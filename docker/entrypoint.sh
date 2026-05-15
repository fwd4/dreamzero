#!/bin/bash
set -euo pipefail

# Code and checkpoints come from the NFS mount at /workspace/dreamzero.
# The Python venv at /opt/dreamzero-venv has all dependencies installed.
# eval_utils is importable because we cd into the NFS repo root first.
cd /workspace/dreamzero

# DIT attention cache accelerates AR inference by ~30% (default: enabled).
# Set ENABLE_DIT_CACHE=false to disable.
DIT_CACHE_FLAG="--enable-dit-cache"
if [ "${ENABLE_DIT_CACHE:-true}" = "false" ]; then
    DIT_CACHE_FLAG="--no-enable-dit-cache"
fi

# NO_FRAME_BUFFER (any non-empty value) → pass --no-frame-buffer to the
# server so it skips temporal frame accumulation and infers from the
# single latest frame each call. Used for the no-history ablation; the
# orchestrator injects this env var at endpoint create time when the
# experiment yaml requests it.
NO_FB_FLAG=""
if [ -n "${NO_FRAME_BUFFER:-}" ]; then
    NO_FB_FLAG="--no-frame-buffer"
fi

# S3 upload of per-session imaginary mp4s is wired through the server's
# _maybe_init_s3 (socket_test_optimized_AR.py) and activates automatically
# when STORAGE_BUCKET, STORAGE_PREFIX, AWS_ACCESS_KEY_ID, and
# AWS_SECRET_ACCESS_KEY are all set in the environment. No flag here.

# SIM_CHUNK_DURATION_S (optional, seconds, default unset → fps=15 legacy):
# wall-clock duration of one client-side ``open_loop_horizon`` sim chunk
# (= open_loop_horizon / sim_fps). When set, the server picks the
# imaginary mp4 encode fps so its wall-clock duration matches the sim
# mp4's — required for the LeRobot per-column-fps dataset path.
# Robolab's DZPipelineClient default is open_loop_horizon=24 with
# sim_fps=15 → SIM_CHUNK_DURATION_S=1.6.

exec /opt/dreamzero-venv/bin/python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${NPROC_PER_NODE:-2} \
    socket_test_optimized_AR.py \
    --port ${SERVER_PORT:-8000} \
    --model-path ${MODEL_PATH:-/workspace/dreamzero/checkpoints/DreamZero-DROID} \
    ${DIT_CACHE_FLAG} \
    ${NO_FB_FLAG} \
    "$@"
