import logging
import os
import typing

import torch

LOGGER = logging.getLogger(__name__)


def save_checkpoint(model, fpath: os.PathLike):
    LOGGER.info(f"Saving checkpoint to: {fpath}")
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, fpath)


def save_checkpoint_with_config(model: torch.nn.Module, config, fpath: os.PathLike):
    LOGGER.info(f"Saving checkpoint to: {fpath}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "class": model.__class__,
        "config": config,
    }
    torch.save(checkpoint, fpath)


def load_checkpoint_with_config(fpath: os.PathLike, device=torch.device("cpu")
                                ) -> tuple[torch.nn.Module, typing.Any] | dict[str | typing.Any]:
    LOGGER.info(f"Loading model from: {fpath}")
    checkpoint: dict = torch.load(fpath, map_location=device)
    if "config" not in checkpoint or "class" not in checkpoint:
        LOGGER.warning("This checkpoint has no config or model class attribute!"
                       f" Returning checkpoint object instead:\n{checkpoint}")
        return checkpoint
    try:
        model_class: object = checkpoint.get("class")
        model_config = checkpoint.get("config")
        model = model_class.from_config(model_config)
        model.load_state_dict(checkpoint.get("state_dict"))
        LOGGER.info(f"Loaded model of class {model_class.__name__}.")
        return model, model_config
    except AttributeError as e:
        LOGGER.error("Error loading model")
        LOGGER.debug(e)
        exit(1)
