from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_local_global import BackboneLocalGlobal, BackboneLocalGlobalCfg
from .backbone_vggt_omega import BackboneVggtOmega, BackboneVggtOmegaCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    "local_global": BackboneLocalGlobal,
    "vggt_omega": BackboneVggtOmega,
}

BackboneCfg = (
    BackboneResnetCfg
    | BackboneDinoCfg
    | BackboneCrocoCfg
    | BackboneLocalGlobalCfg
    | BackboneVggtOmegaCfg
)


def get_backbone(cfg: BackboneCfg, d_in: int = 3, use_checkpoint: bool=False) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in, use_checkpoint=use_checkpoint)
