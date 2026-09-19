"""Triangle-splat DPT output head.

The head intentionally keeps the GS-DPT parameter contract so checkpoints with
the DA3 Gaussian head can still be loaded.  Two small, newly initialized
channels predict triangle opacity and sigma in addition to the original
``raw_gs`` and ``raw_gs_conf`` outputs.
"""

from typing import Dict as TyDict
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.model.gsdpt import GSDPT
from depth_anything_3.model.utils.head_utils import activate_head_gs, custom_interpolate


class _ExtraOutputHead(nn.Module):
    """Independent opacity and sigma projections from the fused DPT feature."""

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.opacity = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for branch in (self.opacity, self.sigma):
            nn.init.kaiming_normal_(branch[0].weight, mode="fan_out", nonlinearity="relu")
            nn.init.normal_(branch[0].bias, mean=0.0, std=0.02)
            nn.init.normal_(branch[2].weight, mean=0.0, std=0.02)
            nn.init.normal_(branch[2].bias, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.opacity(features), self.sigma(features)), dim=1)

    def __getitem__(self, index: int) -> nn.Module:
        """Keep the old ``extra_output_conv[0]/[2]`` inspection API usable."""
        if index not in (0, 2):
            raise IndexError(index)
        return self.opacity[index]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older checkpoints used one shared trunk and a two-channel final
        # convolution. Split those tensors into the new independent branches.
        old_keys = {
            "0.weight": ("opacity.0.weight", "sigma.0.weight"),
            "0.bias": ("opacity.0.bias", "sigma.0.bias"),
            "2.weight": ("opacity.2.weight", "sigma.2.weight"),
            "2.bias": ("opacity.2.bias", "sigma.2.bias"),
        }
        for old_suffix, new_suffixes in old_keys.items():
            old_key = prefix + old_suffix
            if old_key not in state_dict:
                continue
            old_value = state_dict.pop(old_key)
            if old_suffix.startswith("2."):
                state_dict[prefix + new_suffixes[0]] = old_value[0:1].clone()
                state_dict[prefix + new_suffixes[1]] = old_value[1:2].clone()
            else:
                state_dict[prefix + new_suffixes[0]] = old_value.clone()
                state_dict[prefix + new_suffixes[1]] = old_value.clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _ScaleOutputHead(nn.Module):
    """Independent three-channel scale projection.

    The legacy DA3 GS head still emits scale-shaped channels for checkpoint
    compatibility, but TSDPT geometry must not inherit those pretrained
    values.  Keep the feature trunk useful while zeroing the final projection
    so the initial scale logits are exactly zero.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1, padding=0),
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.net[0].bias, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def differential_normals(point_map: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate camera-space normals from a dense point map.

    This is the geometry-normal path used by TriSplat (central finite
    differences followed by a cross product).  Inputs may be channel-first
    ``[..., 3, H, W]`` or channel-last ``[..., H, W, 3]``; the result follows
    the input layout and is unit-normalized.  Border differences replicate the
    nearest pixel, matching TriSplat's padding behavior.
    """
    if point_map.ndim < 4:
        raise ValueError(f"point_map must have at least 4 dimensions, got {point_map.shape}")
    channel_first = point_map.shape[-3] == 3
    if channel_first:
        points = point_map
    elif point_map.shape[-1] == 3:
        points = point_map.movedim(-1, -3)
    else:
        raise ValueError("point_map must have 3 channels in the last or third-last dimension")

    padded = F.pad(points, (1, 1, 1, 1), mode="replicate")
    dx = padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]
    dy = padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]
    normals = F.normalize(torch.cross(dx, dy, dim=-3), dim=-3, eps=eps)
    return normals if channel_first else normals.movedim(-3, -1)


class TSDPT(GSDPT):
    """GS-DPT-compatible head with triangle scale, opacity, and sigma outputs.

    ``output_dim`` remains the original GS output dimension (38 for the DA3
    GIANT checkpoint).  Opacity and sigma are in ``extra_output_conv`` and
    scale is in the independent zero-output-initialized ``scale_output_conv``;
    neither changes the pretrained ``scratch.output_conv2`` contract.
    The returned tensors are ``[B*S,H,W]`` per view, like :class:`GSDPT`.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 38,
        activation: str = "linear",
        conf_activation: str = "sigmoid",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        conf_dim: int = 1,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
        opacity_activation: str = "linear",
        sigma_activation: str = "linear",
        scale_activation: str = "linear",
    ) -> None:
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            feature_only=feature_only,
            down_ratio=down_ratio,
            conf_dim=conf_dim,
            norm_type=norm_type,
            fusion_block_inplace=fusion_block_inplace,
        )
        self.opacity_activation = opacity_activation
        self.sigma_activation = sigma_activation
        self.scale_activation = scale_activation
        # These channels do not exist in the GSDPT checkpoint. Keep them as
        # independent randomly initialized heads so opacity can use a separate
        # optimizer learning rate from sigma and the legacy GS outputs.
        self.extra_output_conv = _ExtraOutputHead(features // 2)
        # This head is intentionally absent from the DA3 checkpoint.  Its
        # final projection is zero-initialized, so pretrained GS scale logits
        # cannot leak into the direct-difference triangle geometry.
        self.scale_output_conv = _ScaleOutputHead(features // 2)

    @staticmethod
    def _activate(value: torch.Tensor, activation: str) -> torch.Tensor:
        if activation == "linear":
            return value
        if activation == "sigmoid":
            return value.sigmoid()
        if activation == "softplus":
            return F.softplus(value)
        if activation == "exp":
            return value.exp()
        raise ValueError(f"Unknown TSDPT activation: {activation}")

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        images: torch.Tensor,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = self.norm(feats[take_idx][:, patch_start_idx:])
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            resized_feats.append(self.resize_layers[stage_idx](x))

        fused = self._fuse(resized_feats)
        fused = self.scratch.output_conv1(fused)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        fused = fused + self.images_merger(images)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        main_logits = self.scratch.output_conv2(fused)
        pred, conf = activate_head_gs(
            main_logits,
            activation=self.activation,
            conf_activation=self.conf_activation,
            conf_dim=self.conf_dim,
        )
        extra = self.extra_output_conv(fused)
        opacity = self._activate(extra[:, 0], self.opacity_activation)
        sigma = self._activate(extra[:, 1], self.sigma_activation)
        scale = self._activate(self.scale_output_conv(fused), self.scale_activation)
        return {
            self.head_main: pred.squeeze(1),
            f"{self.head_main}_conf": conf.squeeze(1),
            f"{self.head_main}_opacity": opacity,
            f"{self.head_main}_sigma": sigma,
            f"{self.head_main}_scale": scale,
        }
