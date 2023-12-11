import pandas as pd

from representation.src.actions.abstract import AbstractAction
from representation.src.smiles_utils.cleaner import SmilesCleaner


class SmilesCleaning(AbstractAction):
    input_column: str
    output_column: str

    def __init__(
        self,
        input_columns: str,
        output_columns: str = "canonical_smiles",
        verbose=True,
        logger=None,
        sanitize=True,
        remove_salts=True,
        remove_stereo=True,
        remove_metal_atoms=False,
        keep_largest_fragment=True,
        neutralize_mol=False,
        standardize_tautomers=False,
        remove_duplicates=True,
        canonicalize_smiles=True,
        limit_seq_len=None,
        constrains=None,
    ) -> None:
        super().__init__()
        self.input_column = input_columns
        self.output_column = output_columns

        self.smiles_cleaner = SmilesCleaner(
            verbose=verbose,
            logger=logger,
            sanitize=sanitize,
            remove_salts=remove_salts,
            remove_stereo=remove_stereo,
            remove_metal_atoms=remove_metal_atoms,
            keep_largest_fragment=keep_largest_fragment,
            neutralize_mol=neutralize_mol,
            standardize_tautomers=standardize_tautomers,
            remove_duplicates=remove_duplicates,
            canonicalize_smiles=canonicalize_smiles,
            limit_seq_len=limit_seq_len,
            constrains=constrains,
        )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the smiles"""
        cleaned_df = self.smiles_cleaner.clean_data(df, self.input_column)
        cleaned_df = cleaned_df.rename(columns={self.input_column: self.output_column})
        return pd.concat([df, cleaned_df], axis=1)
