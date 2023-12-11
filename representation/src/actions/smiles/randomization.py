import pandas as pd

from representation.src.actions.abstract import AbstractAction
from representation.src.smiles_utils.randomization import randomize_smiles


class SmilesRandomizer(AbstractAction):
    input_column: str
    output_column: str

    def __init__(
        self,
        input_column: str,
        output_column: str = "randomized",
        random_type: str = "restricted",
    ) -> None:
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column

        self.random_type = random_type

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enumerate smiles"""
        randomized = df[[self.input_column]].applymap(lambda x: randomize_smiles(x, self.random_type))
        randomized = randomized.rename(columns={self.input_column: self.output_column})
        return pd.concat([df, randomized], axis=1)
