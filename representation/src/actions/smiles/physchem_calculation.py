import pandas as pd

from representation.src.actions.abstract import AbstractAction
from representation.src.smiles_utils.physchem import calc_physchem


class PhyschemCalculation(AbstractAction):
    input_column: str
    output_column: list[str]

    def __init__(
        self,
        input_column: str,
        output_column: list[str] = [
            "alogp",
            "min_charge",
            "max_charge",
            "val_electrons",
            "hdb",
            "hba",
            "balaban_j",
            "refrac",
            "tpsa",
        ],
    ) -> None:
        super().__init__()
        self.input_column = input_column
        self.output_column = output_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate physico-chemical properties of the smiles."""
        physchem = df[[self.input_column]].applymap(lambda x: calc_physchem(x))
        physchem = pd.DataFrame(
            physchem[self.input_column].to_list(),
            columns=self.output_column,
            index=df.index,
        )
        return pd.concat([df, physchem], axis=1)
