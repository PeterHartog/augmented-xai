from typing import Optional

import pandas as pd

from representation.src.actions.abstract import AbstractAction
from representation.src.smiles_utils.utils import heavy_atoms


class DropAllNA(AbstractAction):
    input_column: Optional[str]
    output_column: Optional[str]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops indices with None of dataset."""
        return df.dropna()


class DropDuplicates(AbstractAction):
    def __init__(self, input_column: list[str]) -> None:
        super().__init__()
        self.input_column = input_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicates of dataset."""
        return df.drop_duplicates(subset=self.input_column)


class MaxSeqLen(AbstractAction):
    def __init__(
        self,
        input_column: str,
        output_column: str = "enumerated",
        max_len: int = 250,
    ) -> None:
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column

        self.max_len = max_len

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove smiles with more than n tokens."""
        smi_filter: pd.Series[bool] = df[self.input_column].apply(lambda x: len(x) <= self.max_len)
        return df.loc[smi_filter]


class MaxHeavyAtoms(AbstractAction):
    def __init__(
        self,
        input_column: str,
        output_column: str = "enumerated",
        max_heavy: int = 30,
    ) -> None:
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column

        self.max_heavy = max_heavy

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove smiles with more than n heavy atoms."""
        smi_filter: pd.Series[bool] = df[self.input_column].apply(lambda x: heavy_atoms(x) <= self.max_heavy)
        return df.loc[smi_filter]
