#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
run_name="${TRISPL_RUN_NAME:-tsdpt_da3_dl3dv_sh_single_scene_resume5000_1k}"
experiment_name="${TRISPL_EXPERIMENT:-trisplat_dl3dv_tsdpt_1k_sh_single_scene}"
checkpoint_load="${TRISPL_CHECKPOINT_LOAD:-/root/data/haoxuan/TriSplat/outputs/exp_tsdpt_da3_dl3dv_5k_noschedule_resume3900_color/2026-08-27_03-27-22/checkpoints/render_step_005000.ckpt}"
max_steps="${TRISPL_MAX_STEPS:-6000}"
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
    --experiment "trisplat_dl3dv_tsdpt_1k_sh_single_scene" \
    --device "${TRISPLAT_RENDER_DEVICE:-0}" \
    --poll-seconds 30 > "/tmp/${run_name}.render_monitor.log" 2>&1 &
fi

cmd=("${python_bin}" -m src.main \
  "+experiment=${experiment_name}" \
  "trainer.max_steps=${max_steps}" "+trainer.devices=8" "+trainer.strategy=ddp" \
  "model.encoder.triangle_scale_max=${triangle_scale_max}" \
  "wandb.mode=disabled" "wandb.name=${run_name}" \
  "dataset.dl3dv.view_sampler.num_context_views=6" \
  "dataset.dl3dv.view_sampler.num_target_views=4" \
  "train.use_mono_normal_teacher=false" "train.eval_model_every_n_val=0" \
  "data_loader.train.batch_size=1" "data_loader.train.num_workers=0" \
  "data_loader.train.persistent_workers=false" "data_loader.val.num_workers=0" \
  "data_loader.val.persistent_workers=false" "checkpointing.every_n_train_steps=100" \
  "checkpointing.load=${checkpoint_load}")
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} ${cmd[*]}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" "${cmd[@]}" 2>&1 | tee "outputs/${run_name}/train.log"
