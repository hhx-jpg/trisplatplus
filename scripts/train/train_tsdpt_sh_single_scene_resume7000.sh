#!/usr/bin/env bash
set -euo pipefail

export TRISPL_RUN_NAME="tsdpt_da3_dl3dv_sh_single_scene_resume7000_1k"
export TRISPL_CHECKPOINT_LOAD="/root/data/haoxuan/TriSplat/outputs/exp_tsdpt_da3_dl3dv_5k_noschedule_resume5000_calibrated/2026-08-27_07-21-46/checkpoints/render_step_007000.ckpt"
export TRISPL_MAX_STEPS="8000"
export TRISPL_TRIANGLE_SCALE_MAX="18.0"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_tsdpt_sh_single_scene.sh"
