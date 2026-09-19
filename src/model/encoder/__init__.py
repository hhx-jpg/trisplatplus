from typing import Optional

from .encoder import Encoder
from .encoder_da3_tsdpt import EncoderDA3TSDPTCfg, EncoderDA3TSDPT

# Keep DA3 forward inspection usable without the optional training-only
# dependencies pulled in by the original TriSplat encoder (LPIPS, Lightning,
# and visualization packages).  The full encoder remains available whenever
# those dependencies are installed.
try:
    from .encoder_trisplat import EncoderTrisplatCfg, EncoderTrisplat
except ModuleNotFoundError:
    EncoderTrisplatCfg = None
    EncoderTrisplat = None
try:
    from .visualization.encoder_visualizer import EncoderVisualizer
except ModuleNotFoundError:
    EncoderVisualizer = None

ENCODERS = {
    **({"trisplat": (EncoderTrisplat, None)} if EncoderTrisplat is not None else {}),
    "da3_tsdpt": (EncoderDA3TSDPT, None),
}

EncoderCfg = (EncoderTrisplatCfg | EncoderDA3TSDPTCfg) if EncoderTrisplatCfg is not None else EncoderDA3TSDPTCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
