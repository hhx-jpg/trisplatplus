import re
from pathlib import Path
from typing import Any

import torch


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file as torch_load_file

        return torch_load_file(path, device="cpu")

    return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError("Unsupported checkpoint format: expected a state_dict or Lightning checkpoint.")


def checkpoint_has_training_state(checkpoint: Any) -> bool:
    if not isinstance(checkpoint, dict):
        return False

    # Lightning still writes ``epoch``, ``global_step`` and ``loops`` when
    # ModelCheckpoint.save_weights_only=True.  Those metadata keys are not
    # sufficient for a true optimizer/scheduler resume and passing such a
    # checkpoint to ``trainer.fit(ckpt_path=...)`` raises at restore time.
    optimizer_states = checkpoint.get("optimizer_states")
    scheduler_states = checkpoint.get("lr_schedulers")
    return bool(optimizer_states) or bool(scheduler_states)


DEFAULT_WEIGHT_ONLY_SCHEDULE_STEP = 200000


def get_checkpoint_schedule_step(checkpoint: Any, path: Path) -> int:
    if isinstance(checkpoint, dict) and "global_step" in checkpoint:
        return int(checkpoint["global_step"])

    match = re.search(r"step[_-](\d+)", path.name)
    if match is None:
        return DEFAULT_WEIGHT_ONLY_SCHEDULE_STEP

    return int(match.group(1))
