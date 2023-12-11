import warnings
from ast import literal_eval
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.registries import AIDD
from tdc.single_pred import Tox
from torch.utils.data import DataLoader, Dataset

from representation.src.datasets.collate_functions.abstract import AbstractCollate
from representation.src.datasets.collate_functions.mask_collate import MaskedCollate
from representation.src.datasets.collate_functions.seq_collate import SeqCollate
from representation.src.datasets.random_seq2seq import RandomSeq2seqDataset
from representation.src.datasets.seq2seq import Seq2seqDataset
from representation.src.tokenizer.abstract import AbstractTokenizer
from representation.src.utils.inspect_kwargs import set_kwargs

SPLIT_OPTIONS = ["random", "scaffold"]
RANDOM_TYPE_OPTIONS = ["unrestricted", "restricted"]


@AIDD.DataRegistry.register_arguments(key="tdc_ames")
@dataclass
class TDCommonsArgs:
    name: str = "tdc_ames"

    split: str = "scaffold"
    use_raw: bool = False
    original_smiles: bool = False
    enumerate: bool = False
    physchem: bool = False

    random_state: int = 1234
    limit: Optional[int] = None
    batch_size: int = 128
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    drop_last: bool = False

    batch_first: bool = True
    pad_to_max: bool = False
    mask_sequences: bool = False
    randomize_smiles: bool = False
    random_type: str = "unrestricted"
    override_pred_split: Optional[str] = None

    root: str = ""

    def __post_init__(self):
        if self.split not in SPLIT_OPTIONS:
            raise Exception(f"{self.split} not in {SPLIT_OPTIONS}")
        if self.random_type not in RANDOM_TYPE_OPTIONS:
            raise Exception(f"{self.random_type} not in {RANDOM_TYPE_OPTIONS}")
        if self.use_raw:
            warnings.warn("Using raw data, setting enum, physchem and original smiles to false")
            self.original_smiles, self.enumerate, self.physchem = False, False, False


@AIDD.DataRegistry.register(key="tdc_ames")
class TDCommonsDatamodule(pl.LightningDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        args = set_kwargs(TDCommonsArgs, **kwargs)

        self.root = args.root

        self.use_raw = args.use_raw
        self.enumerate = args.enumerate
        self.override_pred_split = args.override_pred_split

        self.smiles_col = (
            "original_smiles"
            if args.original_smiles and not args.enumerate
            else "enumerated"
            if args.enumerate
            else "canonical_smiles"
        )
        self.usecols: list[str] = ["Drug_ID", "Y"] + [self.smiles_col]
        self.mapping: dict[str, Union[str, list[str]]] = {"id": "Drug_ID", "label": "Y"}

        if args.physchem:
            self.aux_cols: list[str] = [
                "alogp",
                "min_charge",
                "max_charge",
                "val_electrons",
                "hdb",
                "hba",
                "balaban_j",
                "refrac",
                "tpsa",
            ]
            self.mapping["physchem"] = self.aux_cols
            self.usecols += self.aux_cols

        self.split = args.split

        self.limit = args.limit
        self.train_limit = int(self.limit * 0.8) if self.limit else None
        self.valid_limit = int(self.limit * 0.1) if self.limit else None
        self.test_limit = int(self.limit * 0.1) if self.limit else None

        # Data arguments
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers if torch.cuda.is_available() else 0
        self.persistent_workers = args.persistent_workers if torch.cuda.is_available() else False
        self.pin_memory = args.pin_memory if torch.cuda.is_available() else False
        self.drop_last = args.drop_last if torch.cuda.is_available() else False

        # Collate arguments
        self.batch_first = args.batch_first
        self.pad_to_max = args.pad_to_max
        self.mask_sequences = args.mask_sequences
        self.randomize_smiles = args.randomize_smiles if not self.enumerate else False
        self.random_type = args.random_type

    def set_tokenizer(self, tokenizer: AbstractTokenizer) -> None:
        self.tokenizer = tokenizer

    def prepare_collate_fn(self) -> AbstractCollate:
        max_len = self.tokenizer.max_seq_len if self.pad_to_max else None
        if not self.mask_sequences:
            collate = SeqCollate(batch_first=self.batch_first, pad_idx=self.tokenizer.pad_idx, max_len=max_len)
        else:
            collate = MaskedCollate(
                batch_first=self.batch_first,
                pad_idx=self.tokenizer.pad_idx,
                max_len=max_len,
                msk_idx=self.tokenizer.msk_idx,
                vocab_size=self.tokenizer.vocab_size,
                mask_p=0.15,
                rand_p=0.1,
                unchanged_p=0.1,
            )
        return collate

    def load_data(self, file: str) -> pd.DataFrame:
        return pd.read_csv(file, nrows=self.train_limit, usecols=self.usecols)

    def set_prediction_set(self, dataset: pd.DataFrame):
        self.predict = dataset

    def prepare_data(self):
        self.collate = self.prepare_collate_fn()

        # Load and process TDCommons data here
        if self.use_raw:
            data = Tox(name="AMES")
            split = data.get_split(self.split)
            self.train = split["train"]
            self.valid = split["valid"]
            self.test = split["test"]
            self.predict = None
        else:
            self.train = self.load_data(f"{self.root}/data/ames_{self.split}_train.csv")
            self.valid = self.load_data(f"{self.root}/data/ames_{self.split}_valid.csv")
            self.test = self.load_data(f"{self.root}/data/ames_{self.split}_test.csv")
            self.predict = None

    def setup(self, stage=None):
        self.prepare_data()  # for distributed training

        if stage == "fit" or stage is None:
            if self.enumerate:
                self.train[self.smiles_col] = self.train[self.smiles_col].apply(literal_eval)
                self.valid[self.smiles_col] = self.valid[self.smiles_col].apply(literal_eval)
                self.train = self.train.explode(self.smiles_col, ignore_index=True)
                self.valid = self.valid.explode(self.smiles_col, ignore_index=True)
            self.train_dataset = self.get_dataset(self.train, self.smiles_col, self.mapping)
            self.val_dataset = self.get_dataset(self.valid, self.smiles_col, self.mapping)
        if stage == "test" or stage is None:
            if self.enumerate:
                self.test[self.smiles_col] = self.test[self.smiles_col].apply(literal_eval)
                self.test = self.test.explode(self.smiles_col, ignore_index=True)
            self.test_dataset = self.get_dataset(self.test, self.smiles_col, self.mapping)
        if stage == "predict" or stage is None:
            if self.predict is not None and self.override_pred_split is None:
                self.predict_dataset = self.get_dataset(self.predict, self.smiles_col, self.mapping)
            elif self.predict is None and self.override_pred_split == "train":
                warnings.warn("No prediction set detected, using test set.")
                if self.enumerate:
                    self.train[self.smiles_col] = self.train[self.smiles_col].apply(literal_eval)
                    self.train = self.train.explode(self.smiles_col, ignore_index=True)
                self.predict_dataset = self.get_dataset(self.train, self.smiles_col, self.mapping)
            elif self.predict is None and self.override_pred_split == "valid":
                warnings.warn("No prediction set detected, using test set.")
                if self.enumerate:
                    self.valid[self.smiles_col] = self.valid[self.smiles_col].apply(literal_eval)
                    self.valid = self.valid.explode(self.smiles_col, ignore_index=True)
                self.predict_dataset = self.get_dataset(self.valid, self.smiles_col, self.mapping)
            else:
                warnings.warn("No prediction set detected, using test set.")
                if self.enumerate:
                    self.test[self.smiles_col] = self.test[self.smiles_col].apply(literal_eval)
                    self.test = self.test.explode(self.smiles_col, ignore_index=True)
                self.predict_dataset = self.get_dataset(self.test, self.smiles_col, self.mapping)

    def get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate.collate_fn,  # type: ignore
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, False)

    def predict_dataloader(self):
        return self.get_dataloader(self.predict_dataset, False)

    def get_dataset(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        mapping: dict[str, Union[str, list[str]]],
    ) -> Dataset:
        if self.randomize_smiles:
            return RandomSeq2seqDataset(df, smiles_col, mapping, self.tokenizer, random_type=self.random_type)
        else:
            return Seq2seqDataset(df, smiles_col, mapping, self.tokenizer)
