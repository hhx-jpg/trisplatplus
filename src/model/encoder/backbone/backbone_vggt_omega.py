from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from ..layers.pos_embed import PositionGetter, RoPE2D
from .backbone import Backbone, BackboneOutput


@dataclass
class BackboneVggtOmegaCfg:
    name: Literal["vggt_omega"]
    source_path: str = "/home/v-hanhaoxuan/vggt-omega"
    checkpoint_path: str = "pretrained_weights/vggt_omega_1b_512.pt"
    checkpoint_sha256: str = ""
    trisplat_head_checkpoint_path: str = ""
    frozen: bool = True
    cached_layer_idx: int = 23


class BackboneVggtOmega(Backbone[BackboneVggtOmegaCfg]):
    patch_size = 16
    output_dim = 2048
    patch_start_idx = 17

    def __init__(
        self,
        cfg: BackboneVggtOmegaCfg,
        d_in: int,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__(cfg)
        if d_in != 3:
            raise ValueError(f"VGGT-Omega expects RGB input, got d_in={d_in}")
        if use_checkpoint and not cfg.frozen:
            raise NotImplementedError(
                "Unfrozen VGGT-Omega requires block-level activation checkpointing."
            )
        if cfg.cached_layer_idx != 23:
            raise ValueError("The feature-only adapter currently requires cached_layer_idx=23.")

        source_path = Path(cfg.source_path).expanduser().resolve()
        try:
            from vggt_omega.models.aggregator import Aggregator
        except ImportError as exc:
            raise ImportError(
                "VGGT-Omega is not importable. Install the pinned external dependency with "
                f"`pip install -e {source_path}`."
            ) from exc

        self.aggregator = Aggregator(cached_layer_indices=(cfg.cached_layer_idx,))
        self.position_getter = PositionGetter()
        self.head_rope = RoPE2D(freq=100.0)
        self.register_buffer(
            "_checkpoint_loaded",
            torch.tensor(False),
            persistent=True,
        )

        if cfg.frozen:
            self.aggregator.requires_grad_(False)

    def train(self, mode: bool = True) -> "BackboneVggtOmega":
        super().train(mode)
        if self.cfg.frozen:
            self.aggregator.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
    ) -> BackboneOutput:
        del intrinsics
        if not bool(self._checkpoint_loaded.item()):
            raise RuntimeError(
                "VGGT-Omega weights are not loaded. Use the two-source checkpoint loader "
                "before training or inference."
            )
        if images.ndim != 5:
            raise ValueError(
                "VGGT-Omega images must have shape [B,V,3,H,W], "
                f"got {tuple(images.shape)}"
            )
        batch, views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"VGGT-Omega expects 3 image channels, got {channels}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "VGGT-Omega requires H and W divisible by 16 to avoid silent border "
                f"cropping, got H={height}, W={width}."
            )

        grad_context = torch.no_grad() if self.cfg.frozen else torch.enable_grad()
        with grad_context:
            cached_outputs, patch_start_idx = self.aggregator(images)
        tokens = cached_outputs[self.cfg.cached_layer_idx]
        if tokens is None:
            raise RuntimeError(
                f"VGGT-Omega cache layer {self.cfg.cached_layer_idx} returned no tokens."
            ) 
        if patch_start_idx != self.patch_start_idx:
            raise RuntimeError(
                "Unexpected VGGT-Omega prefix length: "
                f"expected {self.patch_start_idx}, got {patch_start_idx}."
            )

        tokens = tokens.reshape(batch * views, tokens.shape[2], tokens.shape[3])
        patch_positions = self.position_getter(
            batch * views,
            height // self.patch_size,
            width // self.patch_size,
            images.device,
        )
        patch_positions = patch_positions + 1
        prefix_positions = torch.zeros(
            batch * views,
            patch_start_idx,
            2,
            dtype=patch_positions.dtype,
            device=patch_positions.device,
        )
        positions = torch.cat([prefix_positions, patch_positions], dim=1)

        return BackboneOutput(
            tokens=tokens,
            positions=positions,
            patch_start_idx=patch_start_idx,
            intrinsic_pred=None,
        )

    def freeze_modules(self, target: str) -> list[nn.Module | nn.Parameter]:
        if target in {"encoder", "decoder", "encoder+decoder"}:
            return [self.aggregator]
        return super().freeze_modules(target)
