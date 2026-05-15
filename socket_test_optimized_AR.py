import dataclasses
import io
import logging
import socket
import asyncio
import os
import http
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import torch
import tyro
from einops import rearrange
import datetime

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

# Optional S3 upload (only active when AWS_* env vars are set in the
# container). Used by ARDroidRoboarenaPolicy._save_session_video to ship
# imagined videos to s3://{STORAGE_BUCKET}/{STORAGE_PREFIX}/{run_id}/...
# alongside cosmos3's per-task/per-env layout.
try:
    import boto3  # noqa: PLC0415
    _BOTO3_OK = True
except ImportError:
    _BOTO3_OK = False

from openpi_client import base_policy as _base_policy
import websockets.asyncio.server as _server  # noqa: F401 — used by _health_check signature
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "./checkpoints/dreamzero"
    enable_dit_cache: bool = True
    num_dit_steps: int = 8
    num_inference_steps: int | None = None
    profile_inference: bool = False
    index: int = 0
    max_chunk_size: int | None = None  # If None, use config value. Otherwise override max_chunk_size for inference.
    no_frame_buffer: bool = False  # If True, skip temporal frame accumulation and always infer from the single latest frame.


@dataclasses.dataclass
class _SessionState:
    """Per-session inference state for ARDroidRoboarenaPolicy.

    Each unique session_id (one per parallel env) gets its own frame buffers
    and call counter so that interleaved requests from different envs never
    corrupt each other's temporal context.
    """
    frame_buffers: dict = dataclasses.field(default_factory=lambda: {
        "video.exterior_image_1_left": [],
        "video.exterior_image_2_left": [],
        "video.wrist_image_left": [],
    })
    call_count: int = 0
    video_across_time: list = dataclasses.field(default_factory=list)
    # Captured from the first obs in the session so we can build the
    # canonical S3 key when saving the imaginary video at evict time.
    episode_id: str = ""   # e.g. "FoodPacking1CansTask/Run0Env0"
    run_id: str = ""       # client's run_id uuid


# Module-level S3 client + upload pool, created lazily once env vars are seen.
_S3_CLIENT = None
_S3_POOL: ThreadPoolExecutor | None = None
_S3_BUCKET: str | None = None
_S3_PREFIX: str | None = None


def _maybe_init_s3() -> None:
    """Idempotent S3 client init from AWS_* + STORAGE_{BUCKET,PREFIX} env vars."""
    global _S3_CLIENT, _S3_POOL, _S3_BUCKET, _S3_PREFIX
    if _S3_CLIENT is not None:
        return
    if not _BOTO3_OK:
        return
    bucket = os.environ.get("STORAGE_BUCKET")
    prefix = os.environ.get("STORAGE_PREFIX")
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (bucket and prefix and ak and sk):
        logger.info(
            "[s3] imaginary-mp4 upload disabled "
            f"(bucket={bool(bucket)} prefix={bool(prefix)} ak={bool(ak)})"
        )
        return
    try:
        _S3_CLIENT = boto3.client(
            "s3",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        _S3_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="s3-imag")
        _S3_BUCKET = bucket
        _S3_PREFIX = prefix
        logger.info(f"[s3] imaginary-mp4 upload enabled -> s3://{bucket}/{prefix}/")
    except Exception as e:
        logger.warning(f"[s3] init failed: {e}")


def _upload_imaginary_to_s3(mp4_bytes: bytes, key: str, episode_id: str) -> None:
    """Worker-thread S3 PUT for one imaginary mp4. Errors are logged, not raised."""
    try:
        t0 = time.monotonic()
        _S3_CLIENT.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=mp4_bytes,
            ContentType="video/mp4",
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"[s3] uploaded imaginary mp4 -> s3://{_S3_BUCKET}/{key} "
            f"({len(mp4_bytes)/1024:.1f} KB, {elapsed_ms:.0f} ms, {episode_id})"
        )
    except Exception as e:
        logger.warning(f"[s3] upload failed for {key}: {e}")


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.

    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Per-session frame accumulation (each parallel env has isolated buffers)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination

    Frame-buffer design
    -------------------
    State is keyed by session_id so that N parallel envs can share one server
    without their frame histories contaminating each other.

    Padding policy: we enter multi-frame mode only once a session has
    accumulated FRAMES_PER_CHUNK genuine frames.  Until then we send a single
    (latest) frame — the model resets its KV cache on 1-frame input, which is
    the same behaviour as the previous global-state implementation and avoids
    feeding out-of-distribution repeated-frame padding during episode warmup.

    KV-cache note: the underlying WANPolicyHead stores current_start_frame and
    KV tensors as single shared attributes.  Interleaved sessions therefore
    still cross-contaminate the KV cache; full per-session KV isolation would
    require save/restore of those tensors and is left as future work.
    """

    # Number of genuine frames required before switching to multi-frame mode.
    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        no_frame_buffer: bool = False,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir
        self._no_frame_buffer = no_frame_buffer

        # Per-session state: keyed by session_id string.
        self._sessions: dict[str, _SessionState] = {}
        self._msg_index = 0

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _convert_observation(self, obs: dict, session: _SessionState) -> dict:
        """Convert roboarena observation format to AR_droid format.

        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str

        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}

        # Map image keys (roboarena uses 0-indexed, AR_droid uses 1-indexed)
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }

        if self._no_frame_buffer:
            # No temporal accumulation: always infer from the single latest frame only.
            for roboarena_key, droid_key in image_key_mapping.items():
                if roboarena_key in obs:
                    data = obs[roboarena_key]
                    if isinstance(data, np.ndarray):
                        frame = data[-1] if data.ndim == 4 else data
                        converted[droid_key] = frame[np.newaxis]  # (1, H, W, 3)
        else:
            # Append incoming frame(s) to per-session buffers
            for roboarena_key, droid_key in image_key_mapping.items():
                if roboarena_key in obs:
                    data = obs[roboarena_key]
                    if isinstance(data, np.ndarray):
                        if data.ndim == 4:
                            session.frame_buffers[droid_key].extend(list(data))
                        else:
                            session.frame_buffers[droid_key].append(data)

            # Switch to multi-frame mode only once every camera has FRAMES_PER_CHUNK
            # genuine frames.  Before that, send the single latest frame so the model
            # resets cleanly rather than receiving out-of-distribution padding.
            min_buf = min(len(b) for b in session.frame_buffers.values())
            num_frames = self.FRAMES_PER_CHUNK if min_buf >= self.FRAMES_PER_CHUNK else 1

            # Build video tensors and keep buffers bounded
            for droid_key, buffer in session.frame_buffers.items():
                if buffer:
                    frames_to_use = buffer[-num_frames:]
                    converted[droid_key] = np.stack(frames_to_use, axis=0)
                    # Trim to at most FRAMES_PER_CHUNK so memory stays bounded
                    if len(buffer) > self.FRAMES_PER_CHUNK:
                        session.frame_buffers[droid_key] = buffer[-self.FRAMES_PER_CHUNK:]

        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""

        return converted
    
    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.
        
        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        
        return action
    
    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle
        
        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)
        
        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        
        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)
    
    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations.

        Args:
            obs: Observation dict in roboarena format

        Returns:
            action: (N, 8) action array
        """
        session_id = obs.get("session_id") or "__default__"

        # Get or create per-session state — no global reset on session change
        if session_id not in self._sessions:
            logger.info(f"New session: '{session_id}'")
            self._sessions[session_id] = _SessionState()
        session = self._sessions[session_id]
        # Capture episode_id / run_id from every request that carries them.
        # The S3 key built at session-evict needs both. Idempotent overwrite
        # to the most-recent non-empty value rescues a late-stamped request
        # if the first one was missing the fields (e.g. older client, patch
        # run, retry). Skip empty values so a single later malformed request
        # doesn't clear the captured ids.
        ep = obs.get("episode_id")
        if ep:
            session.episode_id = ep
        rid = obs.get("run_id")
        if rid:
            session.run_id = rid

        self._msg_index += 1
        session.call_count += 1

        # Convert observation using per-session buffers
        converted_obs = self._convert_observation(obs, session)

        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # Broadcast obs to workers
        self._broadcast_batch_to_workers(converted_obs)

        # Distributed forward pass
        batch = Batch(obs=converted_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        session.video_across_time.append(video_pred)

        # Extract and convert action
        action_chunk_dict = result_batch.act
        action_dict = {k: getattr(action_chunk_dict, k) for k in dir(action_chunk_dict) if k.startswith("action.")}
        return self._convert_action(action_dict)

    def _save_session_video(self, session: _SessionState) -> None:
        """Decode the imaginary video accumulated for one session and ship it.

        Writes locally (legacy) when ``self._output_dir`` is set AND uploads
        to S3 when ``_S3_CLIENT`` is configured. The S3 path mirrors cosmos3's
        per-(task, run, env) layout:

            s3://{STORAGE_BUCKET}/{STORAGE_PREFIX}/{run_id}/{episode_id}/
                imaginary/imaginary_episode.mp4
        """
        if not session.video_across_time:
            return
        if not self._output_dir and _S3_CLIENT is None:
            return
        try:
            video_across_time_cat = torch.cat(session.video_across_time, dim=2)
            frames = self._policy.trained_model.action_head.vae.decode(
                video_across_time_cat,
                tiled=self._policy.trained_model.action_head.tiled,
                tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            frame_list = list(frames)
            if not (frame_list and len(frame_list[0].shape) == 3 and frame_list[0].shape[2] in [1, 3, 4]):
                return
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            n_blocks = (len(frame_list) - 1) // 8

            # Pick the encode fps so the imaginary mp4's wall-clock duration
            # equals the simulator's wall-clock duration for the same
            # episode. The world model emits ~4 imaginary frames per
            # ``open_loop_horizon``-step sim chunk, so a fixed ``fps=15``
            # would play ~6× faster than sim — useless for side-by-side
            # comparison or a LeRobot dataset that wants matching episode
            # lengths across video columns.
            #
            #   fps_imaginary = total_frames / (num_chunks × chunk_duration_s)
            #
            # where ``chunk_duration_s = open_loop_horizon / sim_fps`` is a
            # property of the client-side eval loop, not the model. The
            # orchestrator (or operator) sets ``SIM_CHUNK_DURATION_S`` to
            # match the client's config (default 1.6s = 24 sim steps at
            # 15 Hz). Falls back to a fixed 15 fps when the env var is
            # absent so legacy local-dev runs still produce something.
            num_chunks = len(session.video_across_time)
            chunk_duration_s_env = os.environ.get("SIM_CHUNK_DURATION_S")
            if chunk_duration_s_env and num_chunks > 0:
                try:
                    chunk_duration_s = float(chunk_duration_s_env)
                    target_fps = max(1.0, len(frame_list) / (num_chunks * chunk_duration_s))
                except (ValueError, ZeroDivisionError):
                    target_fps = 15.0
            else:
                target_fps = 15.0
            logger.info(
                f"[s3] encoding imaginary mp4 at fps={target_fps:.2f} "
                f"({len(frame_list)} frames / {num_chunks} chunks; "
                f"SIM_CHUNK_DURATION_S={chunk_duration_s_env or 'unset → fps=15'})"
            )

            # Encode once into a buffer; reuse for both local + S3 paths.
            mp4_buf = io.BytesIO()
            # imageio writes via a file-like with a name to pick the format.
            mp4_buf.name = "imaginary_episode.mp4"
            imageio.mimsave(mp4_buf, frame_list, fps=target_fps, codec="libx264", format="mp4")
            mp4_bytes = mp4_buf.getvalue()

            # Local mp4 save is a debug convenience. In production (S3
            # configured) the container's ephemeral disk fills up with
            # duplicates of what's already in S3 — skip the write unless
            # the operator explicitly asks for it via ``LOCAL_MP4_SAVE=true``.
            _local_save_forced = os.environ.get("LOCAL_MP4_SAVE", "").lower() == "true"
            _local_save = self._output_dir and (
                _S3_CLIENT is None or _local_save_forced
            )
            if _local_save:
                save_dir = self._output_dir
                os.makedirs(save_dir, exist_ok=True)
                n_existing = len([f for f in os.listdir(save_dir) if f.endswith(".mp4")])
                output_path = os.path.join(save_dir, f"{n_existing:06}_{timestamp}_n{n_blocks}.mp4")
                with open(output_path, "wb") as f:
                    f.write(mp4_bytes)
                logger.info(f"Saved imaginary video locally: {output_path}")

            if _S3_CLIENT is not None and _S3_POOL is not None and session.episode_id and session.run_id:
                key = f"{_S3_PREFIX}/{session.run_id}/{session.episode_id}/imaginary/imaginary_episode.mp4"
                _S3_POOL.submit(_upload_imaginary_to_s3, mp4_bytes, key, session.episode_id)
            elif _S3_CLIENT is not None:
                logger.warning(
                    f"[s3] missing episode_id/run_id on session — skipping upload "
                    f"(episode_id={session.episode_id!r} run_id={session.run_id!r})"
                )
        except Exception as e:
            logger.warning(f"Failed to save imaginary video: {e}")

    def _reset_state(self, session_ids: list[str] | None = None, save_video: bool = True) -> None:
        """Evict one or more sessions, optionally saving their imaginary videos.

        Args:
            session_ids: Sessions to evict.  None evicts all active sessions.
            save_video: Decode and save the accumulated video before eviction.
        """
        targets = session_ids if session_ids is not None else list(self._sessions.keys())
        for sid in targets:
            session = self._sessions.pop(sid, None)
            if session is None:
                continue
            if save_video:
                self._save_session_video(session)
            logger.info(f"Session '{sid}' evicted (call_count={session.call_count}).")

    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.

        The client may pass a ``session_ids`` list to target specific sessions;
        if absent, all active sessions are evicted.
        """
        session_ids = reset_info.get("session_ids", None)
        self._reset_state(session_ids=session_ids, save_video=True)


class _DistributedWorker:
    """Worker harness for non-rank-0 ranks.

    Rank 0 runs the production ``RoboarenaServer`` (from
    ``eval_utils.policy_server``); the other ranks just need to participate
    in the distributed forward pass via ``dist.broadcast`` / ``dist.barrier``.
    This class owns that loop and the obs-broadcast unpickling.

    History: this used to be a full ``WebsocketPolicyServer`` with its own
    ``_handler`` + per-10-chunk mp4 dump path. That handler was never
    actually reached in production (rank 0 has used ``RoboarenaServer``
    since the per-session-frame-buffers refactor); it carried ~290 lines
    of duplicated VAE-decode + mp4-encode logic. Removed.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._signal_group = signal_group

    async def _worker_loop(self):
        """Worker loop for non-rank-0 processes to participate in distributed inference."""
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                # Wait for obs broadcast from rank 0
                # Create a dummy obs dict structure - will be filled by broadcast
                # obs = {}

                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break

                # --- ADD THIS ELIF BLOCK ---
                elif signal == 2:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    # Loop back to the top and wait for the next signal
                    continue

                # Receive the batch data via broadcast/gather mechanism
                # This is a simplified version - the actual obs structure needs to be broadcasted
                batch = self._receive_batch_from_rank0()
                # Participate in distributed forward pass
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        """Receive batch data from rank 0 using torch.distributed primitives."""
        import pickle

        # Receive the size of the pickled data first
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        # Receive the actual data
        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        # Deserialize
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)



def init_mesh() -> DeviceMesh:
    # env vars set by torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def main(args: Args) -> None:
    # Set environment variable for DIT cache.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["NUM_DIT_STEPS"] = str(args.num_dit_steps)
    os.environ["DREAMZERO_PROFILE"] = "true" if args.profile_inference else "false"
    if args.num_inference_steps is not None:
        os.environ["NUM_INFERENCE_STEPS"] = str(args.num_inference_steps)

    # Use TE cuDNN backend for attention.
    os.environ["ATTENTION_BACKEND"] = "TE"

    # Increase the recompile limit to 100 for inference due
    # to autoregressive nature of the model (several possible shapes).
    torch._dynamo.config.recompile_limit = 800

    # Wire up S3 upload for imaginary mp4s (idempotent; no-op if env vars
    # absent so local dev keeps working).
    _maybe_init_s3()

    embodiment_tag = "oxe_droid"
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    # Create server for all ranks - rank 0 handles websocket, others run worker loop
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        # Create output directory for videos
        # Extract parent directory and checkpoint name from model_path
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")
    
    # Create wrapper policy that converts between roboarena and AR_droid formats
    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
        no_frame_buffer=args.no_frame_buffer,
    )
    
    # Configure server for AR_droid (2 external cameras, wrist camera, joint position actions)
    server_config = PolicyServerConfig(
        image_resolution=(180, 320),  # AR_droid expects 180x320 images
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # Track session to reset state for new clients
        action_space="joint_position",
    )
    
    if rank == 0:
        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0 ranks just participate in the distributed forward pass.
        # Rank 0 owns the WebSocket via RoboarenaServer above; workers stay
        # in the broadcast/barrier loop until the signal_group says stop.
        worker = _DistributedWorker(policy=policy, signal_group=signal_group)
        asyncio.run(worker._worker_loop())
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
