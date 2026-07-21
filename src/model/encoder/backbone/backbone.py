from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

T = TypeVar("T")


@dataclass
class BackboneOutput:
    tokens: Float[Tensor, "batch_view token feature"]
    positions: Tensor
    patch_start_idx: int
    intermediate: object | None = None
    intrinsic_pred: Float[Tensor, "batch_view 2"] | None = None


class Backbone(nn.Module, ABC, Generic[T]):
    cfg: T
    patch_size: int
    output_dim: int
    position_getter: object
    head_rope: nn.Module | None

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @property
    def rope(self) -> nn.Module | None:
        """Compatibility alias for TriSplat prediction heads."""
        return self.head_rope

    @abstractmethod
    def forward(
        self,
        images: Float[Tensor, "batch view 3 height width"],
        intrinsics: Float[Tensor, "batch view 3 3"] | None = None,
    ) -> BackboneOutput:
        pass

    def freeze_modules(self, target: str) -> list[nn.Module | nn.Parameter]:
        if target == "encoder+decoder":
            return [self]
        raise ValueError(f"Backbone {type(self).__name__} does not support freeze target: {target}")
