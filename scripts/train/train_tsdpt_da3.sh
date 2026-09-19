#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"; steps=5000; run_name="tsdpt_da3_dl3dv_5k_directdiff"; experiment="trisplat_dl3dv_tsdpt_5k_directdiff"; extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --run-name) run_name="$2"; shift 2 ;;
    --experiment) experiment="$2"; shift 2 ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done
export DL3DV_ROOT="${DL3DV_ROOT:-/root/data/lhmd/dl3dv_torch_960/10K}"
export DA3_CHECKPOINT="${DA3_CHECKPOINT:-/root/data/haoxuan/Depth-Anything-3/checkpoints/DA3-GIANT-1.1}"
export TRISPLAT_LOCAL_LOG_PATH="${TRISPLAT_LOCAL_LOG_PATH:-${REPO_ROOT}/outputs/${run_name}/metrics}"
export PYTHONPATH="${REPO_ROOT}/submodules/diff-triangle-rasterization:${PYTHONPATH:-}"
context_views="${TRISPLAT_CONTEXT_VIEWS:-6}"
python_bin="${TRISPLAT_PYTHON:-/opt/conda/envs/trisplat/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python"
fi
cmd=("${python_bin}" -m src.main "+experiment=${experiment}" "trainer.max_steps=${steps}" "trainer.val_check_interval=500" "+trainer.devices=8" "+trainer.strategy=ddp" "wandb.mode=disabled" "wandb.name=${run_name}" "dataset.dl3dv.view_sampler.num_context_views=${context_views}" "dataset.dl3dv.view_sampler.num_target_views=4" "dataset.dl3dv.input_image_shape=[224,448]" "dataset.dl3dv.original_image_shape=[540,960]" "train.use_mono_normal_teacher=false" "train.eval_model_every_n_val=0" "train.print_log_every_n_steps=10" "data_loader.train.batch_size=1" "data_loader.train.num_workers=0" "data_loader.train.persistent_workers=false" "data_loader.val.num_workers=0" "data_loader.val.persistent_workers=false" "checkpointing.every_n_train_steps=100" "optimizer.backbone_lr_multiplier=0.1")
cmd+=("${extra[@]}")
mkdir -p "outputs/${run_name}"
metrics_dir="outputs/${run_name}/metrics"
dashboard_dir="${metrics_dir}/dashboard"
if [[ "${TRISPLAT_DISABLE_MONITOR:-0}" != "1" ]] && ! pgrep -f "monitor_realtime.py .*${run_name}/metrics/metrics.jsonl" >/dev/null 2>&1; then
  mkdir -p "${dashboard_dir}"
  nohup "${python_bin}" scripts/train/monitor_realtime.py \
    "${metrics_dir}/metrics.jsonl" \
    "${metrics_dir}/renders" \
    "${dashboard_dir}" \
    --interval 30 > "/tmp/${run_name}.monitor.log" 2>&1 &
fi
if [[ "${TRISPLAT_DISABLE_RENDER_MONITOR:-0}" != "1" ]] && ! pgrep -f "render_checkpoints_monitor.py .*--run-name ${run_name}" >/dev/null 2>&1; then
  nohup "${python_bin}" scripts/train/render_checkpoints_monitor.py \
    --run-name "${run_name}" \
    --data-root "${DL3DV_ROOT}" \
    --experiment "${experiment}" \
    --num-context-views "${context_views}" \
    --device "${TRISPLAT_RENDER_DEVICE:-0}" \
    --poll-seconds 30 > "/tmp/${run_name}.render_monitor.log" 2>&1 &
fi
echo "CUDA_VISIBLE_DEVICES=${gpus} ${cmd[*]}"
exec env CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}" 2>&1 | tee "outputs/${run_name}/train.log"
