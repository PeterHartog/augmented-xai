import os
import pathlib
from typing import Any, Optional

import hydra
import pyrootutils
import pytorch_lightning as pl
from aidd_codebase.utils import utils
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

import wandb

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

log = utils.pylogger.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating Tokenizer <{cfg.tokenizer._target_}>")
    tokenizer: Any = hydra.utils.instantiate(cfg.tokenizer)

    log.info(f"Instantiating datamodule <{cfg.DataRegistry._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.DataRegistry)

    if "pretrain_model" in cfg.keys() and cfg.pretrain_model is not None:
        log.info(f"Instantiating model <{cfg.pretrain_model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.pretrain_model)
        ckpt_path = cfg.pretrain_model.checkpoint
    else:
        log.info(f"Instantiating model <{cfg.ModelRegistry._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.ModelRegistry)
        ckpt_path = cfg.get("ckpt_path")

    log.info("Setting tokenizer to model")
    datamodule.set_tokenizer(tokenizer)  # type: ignore
    model.set_tokenizer(tokenizer)  # type: ignore

    if "pretrained_encoder" in cfg.keys() and cfg.pretrained_encoder is not None:
        log.info(f"Instantiating pretrained model <{cfg.pretrained_encoder._target_}>")
        pretrained_model: LightningModule = hydra.utils.instantiate(cfg.pretrained_encoder)
        pretrained_model = pretrained_model.load_from_checkpoint(cfg.pretrained_encoder.checkpoint)
        model.set_encoder(pretrained_model)  # type: ignore
        log.info(f"Loaded pre-trained encoder from {cfg.pretrained_encoder.checkpoint}")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    with open_dict(cfg):
        cfg.data = cfg.DataRegistry
        cfg.model = cfg.ModelRegistry

    object_dict = {
        "cfg": cfg,
        "data": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if trainer.checkpoint_callback is not None and cfg.get("train"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    if cfg.get("predict"):
        log.info("Starting prediction!")
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base=None, config_path="conf/train_config", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    # run the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # finish wandb run
    wandb.finish()

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    from aidd_codebase.framework.initiator import (
        kaiming_initialization,
        xavier_initialization,
    )
    from aidd_codebase.registries import AIDD

    print([registry for registry in AIDD.get_registries().keys()])
    config_check = utils.ConfigChecker(os.path.join(pathlib.Path(__file__).parent.absolute(), "conf"))
    main()
