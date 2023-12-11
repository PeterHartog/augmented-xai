from ast import literal_eval
from typing import Optional

import pandas as pd
from tdc.generation import MolGen


class ChEMBL:
    # Load and process TDCommons data here
    def __init__(
        self,
        root: str,
        smiles_col: str,
        enumerate_data: bool = False,
        train_limit: Optional[int] = None,
        usecols: Optional[list[str]] = None,
    ) -> None:
        self.root = root
        self.enumerate_data = enumerate_data
        self.smiles_col = smiles_col
        self.train_limit = train_limit
        self.usecols = usecols

    def load_data(self, file: str) -> pd.DataFrame:
        data = pd.read_csv(file, nrows=self.train_limit, usecols=self.usecols)
        if self.enumerate_data:
            data[self.smiles_col] = data[self.smiles_col].apply(literal_eval)
            data = data.explode(self.smiles_col, ignore_index=True)
        return data

    def raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = MolGen(name="ChEMBL_V29")
        split = data.get_split()
        train = split["train"]
        valid = split["valid"]
        test = split["test"]
        return train, valid, test

    def full(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = self.load_data(f"{self.root}/data/chembl_train_cleaned.csv")
        valid = self.load_data(f"{self.root}/data/chembl_valid_cleaned.csv")
        test = self.load_data(f"{self.root}/data/chembl_test_cleaned.csv")
        return train, valid, test

    def unique(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = self.load_data(f"{self.root}/data/chembl_train_unique_cleaned.csv")
        valid = self.load_data(f"{self.root}/data/chembl_valid_unique_cleaned.csv")
        test = self.load_data(f"{self.root}/data/chembl_test_unique_cleaned.csv")
        return train, valid, test
