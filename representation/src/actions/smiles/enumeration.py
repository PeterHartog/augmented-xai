from typing import Optional

import pandas as pd

from representation.src.actions.abstract import AbstractAction
from representation.src.smiles_utils.enumeration import enumerate_smi


class SmilesEnumeration(AbstractAction):
    input_column: str
    output_column: str

    def __init__(
        self,
        input_column: str,
        output_column: str = "enumerated",
        n: int = 100,
        limit: Optional[int] = None,
        random_type="unrestricted",
        keep_original: bool = True,
        randomSeed=1234,
    ) -> None:
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column

        self.n = n
        self.limit = limit
        self.random_type = random_type
        self.keep_original = keep_original
        self.random_seed = randomSeed

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enumerate smiles"""
        enumerated = df[[self.input_column]].applymap(
            lambda x: enumerate_smi(x, self.n, self.limit, self.random_type, self.keep_original, self.random_seed)
        )
        enumerated = enumerated.rename(columns={self.input_column: self.output_column})
        return pd.concat([df, enumerated], axis=1)
