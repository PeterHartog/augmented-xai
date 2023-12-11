import hydra
import pandas as pd
import pyrootutils
import pytorch_lightning as pl
from aidd_codebase.utils import utils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = utils.pylogger.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="conf/data_config", config_name="config")
def main(cfg: DictConfig) -> None:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # import dataset
    log.info(f"Importing data <{cfg.load._target_}>")
    df: pd.DataFrame = hydra.utils.instantiate(cfg.load)

    if cfg.save_original and cfg.save_original_path is not None:
        df.to_csv(cfg.save_original_path)

    # run actions
    if "clean" in cfg.keys() and cfg.clean is not None:
        log.info(f"Instantiating Tokenizer <{cfg.clean._target_}>")
        action: AbstractAction = hydra.utils.instantiate(cfg.clean)
        drop_dups = DropDuplicates(input_column=[cfg.clean.output_columns])
        drop_na = DropAllNA()

        df = action.batchify(df, cfg.batch_size)
        df = drop_dups(df)
        df = drop_na(df)

    if "enum" in cfg.keys() and cfg.enum is not None:
        log.info(f"Instantiating Tokenizer <{cfg.enum._target_}>")
        action: AbstractAction = hydra.utils.instantiate(cfg.enum)
        drop_dups = DropDuplicates(input_column=[cfg.clean.output_columns])
        drop_na = DropAllNA()

        df = action.batchify(df, cfg.batch_size)
        df = drop_dups(df)
        df = drop_na(df)

    # save data
    df.to_csv(cfg.save_path)


if __name__ == "__main__":
    from representation.src.actions.abstract import AbstractAction
    from representation.src.actions.smiles.filtering import DropAllNA, DropDuplicates

    main()
