#!/usr/bin/env bash
set -euo pipefail

log_file="${1:?usage: $0 LOG_FILE [INTERVAL_SEC] [STALL_SEC]}"
interval="${2:-30}"
stall_limit="${3:-180}"

while [[ ! -f "$log_file" ]]; do
  printf '[%s] STATUS=WAITING log=%s\n' "$(date '+%F %T')" "$log_file"
  sleep "$interval"
done

while [[ -f "$log_file" ]]; do
  now="$(date +%s)"
  last_line="$(rg 'train step ' "$log_file" | tail -1 || true)"
  last_step="$(printf '%s\n' "$last_line" | sed -n 's/.*train step \([0-9][0-9]*\).*/\1/p')"
  last_time="$(stat -c %Y "$log_file")"
  age=$((now - last_time))
  gpu="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
  printf '[%s] step=%s log_age=%ss' "$(date '+%F %T')" "${last_step:-?}" "$age"
  if (( age >= stall_limit )); then
    printf ' STATUS=STALL'
  else
    printf ' STATUS=RUNNING'
  fi
  printf ' gpu=%s\n' "$gpu"
  sleep "$interval"
done
