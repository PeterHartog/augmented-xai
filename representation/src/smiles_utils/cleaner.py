import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from representation.src.smiles_utils.utils import Converter


class SmilesCleaner:
    def __init__(
        self,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
        sanitize: bool = True,
        remove_salts: bool = True,
        remove_stereo: bool = True,
        remove_metal_atoms: bool = False,
        keep_largest_fragment: bool = True,
        neutralize_mol: bool = False,
        standardize_tautomers: bool = True,
        remove_duplicates: bool = True,
        canonicalize_smiles: bool = True,
        limit_seq_len: Optional[int] = None,
        constrains: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()

        tqdm.pandas()

        self.verbose = verbose
        self.logger = logger

        self.sanitize = sanitize
        self.remove_salts = remove_salts
        self.remove_stereo = remove_stereo
        self.remove_metal_atoms = remove_metal_atoms
        self.keep_largest_fragment = keep_largest_fragment
        self.neutralize_mol = neutralize_mol
        self.standardize_tautomers = standardize_tautomers
        self.canonicalize_smiles = canonicalize_smiles
        self.remove_duplicates = remove_duplicates
        self.limit_seq_len = limit_seq_len
        self.constrains = constrains

        self.original_length: Optional[int] = None
        self.duplicated_length: Optional[int] = None
        self.constrained_length: Optional[int] = None
        self.cleaned_length: Optional[int] = None

    def log(self, msg: str) -> None:
        if not self.verbose:
            pass
        elif self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _remove_salts(self, mols: pd.DataFrame) -> pd.DataFrame:
        def __remove_salts(mol):
            if mol is None:
                return None
            try:
                remover = SaltRemover.SaltRemover()
                return remover.StripMol(mol)
            except ValueError:
                pass
            return mol

        self.log("Stripping salts...")
        mols = mols.progress_applymap(__remove_salts)
        return mols

    def _remove_stereo(self, mols: pd.DataFrame) -> pd.DataFrame:
        def __remove_stereo(mol):
            if mol is None:
                return None
            try:
                Chem.RemoveStereochemistry(mol)
                return mol
            except ValueError:
                pass
            return mol

        self.log("Removing stereochemistry...")
        mols.progress_applymap(__remove_stereo)
        return mols

    def _assign_stereo(self, mols: pd.DataFrame) -> pd.DataFrame:
        def __assign_stereo(mol):
            if mol is None:
                return None
            try:
                Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
                return mol
            except ValueError:
                pass
            return mol

        self.log("Assigning stereochemistry...")
        mols.progress_applymap(__assign_stereo)
        return mols

    def _remove_metal_atoms(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Removing metal atoms...")
        mols = mols.progress_applymap(rdMolStandardize.MetalDisconnector().Disconnect)
        return mols

    def _keep_largest_fragment(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Keeping largest fragment...")
        mols = mols.progress_applymap(rdMolStandardize.FragmentParent)
        return mols

    def _neutralize_mol(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Neutralizing molecules...")
        mols = mols.progress_applymap(rdMolStandardize.Uncharger().uncharge)
        return mols

    def _tautomers(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Enumerating tautomers...")
        mols = mols.progress_applymap(rdMolStandardize.TautomerEnumerator().Canonicalize)
        return mols

    def clean_data(self, df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
        self.log("Converting SMILES to molecules...")
        mols: pd.DataFrame = df[[smiles_column]].progress_applymap(Converter.smile2mol).dropna()

        def sanitize(mol):
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveHs(mol)
                mol = rdMolStandardize.Normalize(mol)
                mol = rdMolStandardize.Reionize(mol)
            except ValueError:
                return None
            return mol

        if self.sanitize:
            self.log("Sanitizing molecules...")
            mols = mols.progress_applymap(sanitize).dropna()

        if self.remove_salts:
            mols = self._remove_salts(mols).dropna()

        if self.remove_stereo:
            mols = self._remove_stereo(mols).dropna()
        else:
            mols = self._assign_stereo(mols).dropna()
            # add mixed data option
            # check non-smile symbols

        if self.remove_metal_atoms:
            mols = self._remove_metal_atoms(mols).dropna()

        if self.keep_largest_fragment:
            mols = self._keep_largest_fragment(mols).dropna()

        if self.neutralize_mol:
            mols = self._neutralize_mol(mols).dropna()

        if self.standardize_tautomers:
            mols = self._tautomers(mols).dropna()

        if self.canonicalize_smiles:
            self.log("Converting molecules to canonical SMILES...")
            cleaned_smiles: pd.DataFrame = (
                mols.progress_applymap(Converter.mol2canonical).replace(r"^\s*$", np.nan, regex=True).dropna()
            )
        else:
            self.log("Converting molecules to SMILES...")
            cleaned_smiles = mols.progress_applymap(Chem.MolToSmiles).replace(r"^\s*$", np.nan, regex=True).dropna()

        if self.remove_duplicates:
            self.log("Removing duplicates...")
            cleaned_smiles = cleaned_smiles.drop_duplicates().dropna()

        if self.limit_seq_len and self.limit_seq_len > 0:
            self.log("Removing SMILES with length > {}...".format(self.limit_seq_len))
            cleaned_smiles = cleaned_smiles[cleaned_smiles.applymap(len) <= self.limit_seq_len].dropna()

        if self.constrains is not None:
            self.log("Applying constrains...")
            for constrain in self.constrains:
                cleaned_smiles = cleaned_smiles[constrain(cleaned_smiles)].dropna()

        return cleaned_smiles
