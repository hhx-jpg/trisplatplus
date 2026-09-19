from dataclasses import dataclass

from jaxtyping import Float
import torch
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Primitives
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    use_valid_mask: bool = False
    # Rendered pixels with almost no accumulated alpha are usually temporary
    # holes while triangle coverage is being learned.  Downweighting them
    # avoids turning a transient hole into a large-loss optimizer skip.
    black_hole_weight: float = 1.0
    black_hole_opacity_threshold: float = 0.05


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Primitives,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        views = batch["context"] if use_context else batch["target"]
        delta = prediction.color - views["image"]
        self.last_metrics = {"mse_black_hole_fraction": torch.zeros((), device=delta.device)}
        pixel_weight = torch.ones(
            delta.shape[:2] + delta.shape[-2:],
            dtype=delta.dtype,
            device=delta.device,
        )
        normalization_weight = pixel_weight
        if (
            self.cfg.black_hole_weight < 1.0
            and prediction.opacity is not None
            and prediction.opacity.shape == pixel_weight.shape
        ):
            hole = prediction.opacity.detach() < self.cfg.black_hole_opacity_threshold
            self.last_metrics["mse_black_hole_fraction"] = hole.float().mean()
            pixel_weight = torch.where(
                hole,
                torch.as_tensor(
                    self.cfg.black_hole_weight,
                    dtype=delta.dtype,
                    device=delta.device,
                ),
                pixel_weight,
            )

        if self.cfg.use_valid_mask and "valid_mask" in views:
            mask = views["valid_mask"].to(dtype=delta.dtype, device=delta.device)
            if mask.ndim == delta.ndim and mask.shape[2] == 1:
                mask = mask[:, :, 0]
            pixel_weight = pixel_weight * mask
            normalization_weight = mask

        weighted = (delta**2) * pixel_weight.unsqueeze(2)
        # Keep the denominator tied to valid image pixels, rather than the
        # downweighted numerator. Otherwise an all-hole view cancels the
        # black-hole weight and still produces the original large loss.
        denom = normalization_weight.sum() * delta.shape[2]
        return self.cfg.weight * weighted.sum() / denom.clamp_min(1.0)
