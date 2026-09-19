#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
run_name="${TRISPL_RUN_NAME:-tsdpt_da3_dl3dv_lpips20_bookshelf_9c_resume7300_1k}"
experiment_name="${TRISPL_EXPERIMENT:-trisplat_dl3dv_tsdpt_1k_lpips20_bookshelf_resume7300}"
checkpoint_load="${TRISPL_CHECKPOINT_LOAD:-/root/data/haoxuan/TriSplat/outputs/exp_tsdpt_da3_dl3dv_5k_noschedule_resume5000_calibrated/2026-08-27_07-21-46/checkpoints/render_step_007300.ckpt}"
max_steps="${TRISPL_MAX_STEPS:-8300}"
triangle_scale_max="${TRISPL_TRIANGLE_SCALE_MAX:-1.25}"
export DL3DV_ROOT="${DL3DV_ROOT:-/root/data/lhmd/dl3dv_torch_960/10K}"
export DA3_CHECKPOINT="${DA3_CHECKPOINT:-/root/data/haoxuan/Depth-Anything-3/checkpoints/DA3-GIANT-1.1}"
export TRISPLAT_LOCAL_LOG_PATH="${TRISPLAT_LOCAL_LOG_PATH:-${REPO_ROOT}/outputs/${run_name}/metrics}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/submodules/diff-triangle-rasterization:${PYTHONPATH:-}"
python_bin="${TRISPLAT_PYTHON:-/opt/conda/envs/trisplat/bin/python}"
if [[ ! -x "${python_bin}" ]]; then python_bin="python"; fi

mkdir -p "outputs/${run_name}/metrics/dashboard"
if [[ "${TRISPLAT_DISABLE_MONITOR:-0}" != "1" ]] && ! pgrep -f "monitor_realtime.py .*${run_name}/metrics/metrics.jsonl" >/dev/null 2>&1; then
  nohup "${python_bin}" scripts/train/monitor_realtime.py \
    "${REPO_ROOT}/outputs/${run_name}/metrics/metrics.jsonl" \
    "${REPO_ROOT}/outputs/${run_name}/metrics/renders" \
    "${REPO_ROOT}/outputs/${run_name}/metrics/dashboard" \
    --interval 30 > "/tmp/${run_name}.monitor.log" 2>&1 &
fi
if [[ "${TRISPLAT_DISABLE_RENDER_MONITOR:-0}" != "1" ]] && ! pgrep -f "render_checkpoints_monitor.py .*--run-name ${run_name}" >/dev/null 2>&1; then
  nohup "${python_bin}" scripts/train/render_checkpoints_monitor.py \
    --run-name "${run_name}" \
    --data-root "${DL3DV_ROOT}" \
    --experiment "${experiment_name}" \
    --device "${TRISPLAT_RENDER_DEVICE:-0}" \
    --poll-seconds 30 > "/tmp/${run_name}.render_monitor.log" 2>&1 &
fi

gpu="${TRISPL_GPU:-0}"
cmd=("${python_bin}" -m src.main \
  "+experiment=${experiment_name}" \
  "trainer.max_steps=${max_steps}" \
  "wandb.mode=disabled" "wandb.name=${run_name}" \
  "dataset.dl3dv.view_sampler.num_context_views=6" \
  "dataset.dl3dv.view_sampler.num_target_views=4" \
  "train.use_mono_normal_teacher=false" "train.eval_model_every_n_val=0" \
  "data_loader.train.batch_size=1" "data_loader.train.num_workers=0" \
  "data_loader.train.persistent_workers=false" "data_loader.val.num_workers=0" \
  "data_loader.val.persistent_workers=false" "checkpointing.every_n_train_steps=100" \
  "checkpointing.load=${checkpoint_load}")
echo "CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
exec env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | tee "outputs/${run_name}/train.log"
