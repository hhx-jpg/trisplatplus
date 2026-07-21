import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file as torch_load_file

        return torch_load_file(path, device="cpu")

    return torch.load(path, map_location="cpu", weights_only=True)


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

    training_state_keys = ("optimizer_states", "lr_schedulers", "loops")
    return any(key in checkpoint for key in training_state_keys)


@dataclass
class CheckpointAudit:
    source: str
    matched: list[str] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)

    def log(self) -> None:
        logger.info(
            "Checkpoint audit for %s: matched=%d converted=%d excluded=%d "
            "missing=%d unexpected=%d shape_mismatches=%d",
            self.source,
            len(self.matched),
            len(self.converted),
            len(self.excluded),
            len(self.missing),
            len(self.unexpected),
            len(self.shape_mismatches),
        )
        if self.converted:
            logger.info("Converted checkpoint keys from %s: %s", self.source, self.converted)
        if self.shape_mismatches:
            logger.warning("Shape mismatches from %s: %s", self.source, self.shape_mismatches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "matched": list(self.matched),
            "converted": list(self.converted),
            "excluded": list(self.excluded),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "shape_mismatches": {
                key: {"source": list(src), "target": list(tgt)}
                for key, (src, tgt) in self.shape_mismatches.items()
            },
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_trisplat_head_warm_start(
    encoder: torch.nn.Module,
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], CheckpointAudit]:
    from src.misc.weight_modify import (
        resize_spatial_conv_kernel,
        resize_spatial_linear_weights,
    )

    source_state = extract_state_dict(load_checkpoint_file(checkpoint_path))
    if any(key.startswith("encoder.") for key in source_state):
        source_state = {
            key.removeprefix("encoder."): value
            for key, value in source_state.items()
            if key.startswith("encoder.")
        }

    target_state = encoder.state_dict()
    audit = CheckpointAudit(source=str(checkpoint_path))
    prepared: dict[str, torch.Tensor] = {}
    spatial_linear_prefixes = ("point_head.proj", "gaussian_head.proj")

    for key, value in source_state.items():
        if key.startswith("backbone."):
            audit.excluded.append(key)
            continue
        if key not in target_state:
            audit.unexpected.append(key)
            continue
        target = target_state[key]
        if value.shape == target.shape:
            prepared[key] = value
            audit.matched.append(key)
            continue

        converted = False
        for prefix in spatial_linear_prefixes:
            if key == f"{prefix}.weight":
                bias_key = f"{prefix}.bias"
                if bias_key not in source_state or bias_key not in target_state:
                    break
                output_channels = 3 if prefix.startswith("point_head") else encoder.raw_gs_dim
                resized_weight, resized_bias = resize_spatial_linear_weights(
                    value,
                    source_state[bias_key],
                    (output_channels, target.shape[0]),
                )
                if resized_weight.shape != target.shape or resized_bias.shape != target_state[bias_key].shape:
                    break
                prepared[key] = resized_weight
                prepared[bias_key] = resized_bias
                audit.converted.extend([key, bias_key])
                converted = True
                break
            if key == f"{prefix}.bias" and key in prepared:
                converted = True
                break
        if converted:
            continue

        if key == "rgb_embed.proj.weight" and value.ndim == 4 and target.ndim == 4:
            resized = resize_spatial_conv_kernel(value, tuple(target.shape[-2:]))
            if resized.shape == target.shape:
                prepared[key] = resized
                audit.converted.append(key)
                continue

        audit.shape_mismatches[key] = (tuple(value.shape), tuple(target.shape))

    audit.missing = sorted(
        key
        for key in target_state
        if not key.startswith("backbone.") and key not in prepared
    )
    audit.matched.sort()
    audit.converted.sort()
    audit.excluded.sort()
    audit.unexpected.sort()
    audit.log()
    return prepared, audit


def load_trisplat_head_warm_start(
    encoder: torch.nn.Module,
    checkpoint_path: Path,
) -> CheckpointAudit:
    prepared, audit = prepare_trisplat_head_warm_start(encoder, checkpoint_path)
    if not prepared:
        raise ValueError(f"No TriSplat head weights matched checkpoint: {checkpoint_path}")
    encoder.load_state_dict(prepared, strict=False)
    return audit


def load_vggt_omega_aggregator(
    backbone: torch.nn.Module,
    checkpoint_path: Path,
    expected_sha256: str = "",
) -> CheckpointAudit:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")
    if expected_sha256:
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                "VGGT-Omega checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )

    state_dict = extract_state_dict(load_checkpoint_file(checkpoint_path))
    aggregator_state = {
        key.removeprefix("aggregator."): value
        for key, value in state_dict.items()
        if key.startswith("aggregator.")
    }
    if not aggregator_state:
        raise ValueError(
            f"No aggregator.* weights found in VGGT-Omega checkpoint: {checkpoint_path}"
        )

    backbone.aggregator.load_state_dict(aggregator_state, strict=True)
    backbone._checkpoint_loaded.fill_(True)
    audit = CheckpointAudit(
        source=str(checkpoint_path),
        matched=sorted(aggregator_state),
        excluded=sorted(key for key in state_dict if not key.startswith("aggregator.")),
    )
    audit.log()
    return audit


DEFAULT_WEIGHT_ONLY_SCHEDULE_STEP = 200000


def get_checkpoint_schedule_step(checkpoint: Any, path: Path) -> int:
    if isinstance(checkpoint, dict) and "global_step" in checkpoint:
        return int(checkpoint["global_step"])

    match = re.search(r"step[_-](\d+)", path.name)
    if match is None:
        return DEFAULT_WEIGHT_ONLY_SCHEDULE_STEP

    return int(match.group(1))


def resolve_omega_checkpoint_paths(
    omega_cfg: Any,
    repo_root: Path,
) -> tuple[Path, Path | None, str]:
    """Resolve Omega checkpoint paths relative to *repo_root*.

    Hydra changes the working directory at runtime, so relative paths in
    YAML configs (e.g. ``pretrained_weights/vggt_omega_1b_512.pt``) must
    be anchored to a fixed base.

    Returns:
        (omega_ckpt, trisplat_head_ckpt_or_none, sha256)
    """
    omega_ckpt = Path(omega_cfg.checkpoint_path)
    if not omega_ckpt.is_absolute():
        omega_ckpt = repo_root / omega_ckpt

    head_path: str = getattr(omega_cfg, "trisplat_head_checkpoint_path", "") or ""
    head_ckpt: Path | None = None
    if head_path:
        head_ckpt = Path(head_path)
        if not head_ckpt.is_absolute():
            head_ckpt = repo_root / head_ckpt

    sha256: str = getattr(omega_cfg, "checkpoint_sha256", "") or ""
    return omega_ckpt.resolve(), head_ckpt.resolve() if head_ckpt else None, sha256


def init_vggt_omega_backbone(
    encoder: torch.nn.Module,
    omega_cfg: Any,
    repo_root: Path,
) -> dict[str, CheckpointAudit]:
    """Shared VGGT-Omega two-source initialisation for training and inference.

    Order:
    1. Optional TriSplat head warm-start (7→8 spatial conversion).
    2. Omega Aggregator strict-load.

    Returns a dict ``{"heads": audit | None, "omega": audit}``.
    """
    omega_ckpt, head_ckpt, sha256 = resolve_omega_checkpoint_paths(
        omega_cfg,
        repo_root,
    )

    audits: dict[str, CheckpointAudit] = {}

    if head_ckpt is not None:
        audits["heads"] = load_trisplat_head_warm_start(encoder, head_ckpt)
    else:
        audits["heads"] = None

    audits["omega"] = load_vggt_omega_aggregator(
        encoder.backbone,
        omega_ckpt,
        sha256,
    )
    return audits
