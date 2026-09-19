import shutil
import json
import os
from pathlib import Path
from typing import Any, Optional

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image

LOG_PATH = Path(os.environ.get("TRISPLAT_LOCAL_LOG_PATH", "outputs/local"))


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.experiment = None
        # Every DDP rank constructs a logger, but only rank zero writes files.
        # Letting all ranks remove the directory races with metric/image writes.
        if os.environ.get("LOCAL_RANK", "0") == "0":
            shutil.rmtree(LOG_PATH, ignore_errors=True)

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        record = {"step": int(step)}
        for key, value in metrics.items():
            try:
                record[key] = float(value)
            except (TypeError, ValueError):
                continue
        # Model-only continuation resets Lightning's logger step, but the
        # model logs its continued schedule position explicitly.
        if "info/global_step" in record:
            record["step"] = int(record["info/global_step"])
        with (LOG_PATH / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = LOG_PATH / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
