import logging
import os

import torch

LOGGER = logging.getLogger(__name__)


def save_checkpoint(model, fpath: os.PathLike):
    LOGGER.info(f"Saving checkpoint to: {fpath}")
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, fpath)
