from typing import Optional

import pandas as pd
from tdc.generation import MolGen
from tdc.single_pred import Tox

from representation.src.actions.smiles.cleaning import SmilesCleaning
from representation.src.actions.smiles.enumeration import SmilesEnumeration
from representation.src.actions.smiles.filtering import DropAllNA, DropDuplicates
from representation.src.actions.smiles.physchem_calculation import PhyschemCalculation


def load_chembl():
    data = MolGen(name="ChEMBL_V29")
    dfs = data.get_split(method="random")
    df = pd.concat({k: dfs[k] for k in ["test", "valid", "train"]}, axis=0)
    df = df.reset_index(level=0, drop=False).rename(columns={"level_0": "split"}).reset_index(drop=True)
    return df


def load_random_ames():
    data = Tox(name="AMES")
    dfs = data.get_split(method="random")
    df = pd.concat({k: dfs[k] for k in ["test", "valid", "train"]}, axis=0)
    df = df.reset_index(level=0, drop=False).rename(columns={"level_0": "split"}).reset_index(drop=True)
    return df


def load_scaffold_ames():
    data = Tox(name="AMES")
    dfs = data.get_split(method="scaffold")
    df = pd.concat({k: dfs[k] for k in ["test", "valid", "train"]}, axis=0)
    df = df.reset_index(level=0, drop=False).rename(columns={"level_0": "split"}).reset_index(drop=True)
    return df


def cleaning_steps(
    df: pd.DataFrame, smiles_column: str, batch_size: Optional[int] = 2048, passi: bool = False
) -> pd.DataFrame:
    clean_action = SmilesCleaning(
        input_columns=smiles_column,
        output_columns="canonical_smiles",
        verbose=False,
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
    )
    enum_action = SmilesEnumeration(
        input_column="canonical_smiles",
        output_column="enumerated",
        n=100,
        limit=10,
        random_type="unrestricted",
        keep_original=True,
        randomSeed=1234,
    )
    unrestricted_action = SmilesEnumeration(
        input_column="canonical_smiles",
        output_column="unrestricted",
        n=100,
        limit=10,
        random_type="unrestricted",
        keep_original=False,
        randomSeed=1234,
    )
    restricted_action = SmilesEnumeration(
        input_column="canonical_smiles",
        output_column="restricted",
        n=100,
        limit=10,
        random_type="restricted",
        keep_original=False,
        randomSeed=1234,
    )
    physchem_action = PhyschemCalculation(input_column="canonical_smiles")
    drop_dups = DropDuplicates(input_column=["canonical_smiles"])
    drop_na = DropAllNA()

    df = clean_action.batchify(df, batch_size=batch_size) if batch_size else clean_action(df)
    df = drop_dups(df)
    df = drop_na(df)

    df = enum_action.batchify(df, batch_size=batch_size) if batch_size else enum_action(df)
    if not passi:
        df = unrestricted_action.batchify(df, batch_size=batch_size) if batch_size else unrestricted_action(df)
        df = restricted_action.batchify(df, batch_size=batch_size) if batch_size else restricted_action(df)
    df = physchem_action.batchify(df, batch_size=batch_size) if batch_size else physchem_action(df)
    df = drop_na(df)

    df = df.rename(columns={smiles_column: "original_smiles"})
    return df


def main(batch_size: int = 5096):
    # Ames Scaffold
    ames_scaffold = load_scaffold_ames()
    cleaned_ames = cleaning_steps(ames_scaffold, "Drug", batch_size)
    cleaned_ames["Drug_ID"] = cleaned_ames["Drug_ID"].apply(lambda x: x.strip("Drug "))
    for name in ["train", "valid", "test"]:
        cleaned_ames[cleaned_ames["split"] == name].to_csv(f"data/ames_scaffold_{name}.csv", index=False)

    # Ames Random
    ames_random = load_random_ames()
    cleaned_random_ames = cleaning_steps(ames_random, "Drug", batch_size)
    cleaned_random_ames["Drug_ID"] = cleaned_random_ames["Drug_ID"].apply(lambda x: x.strip("Drug "))
    for name in ["train", "valid", "test"]:
        cleaned_random_ames[cleaned_random_ames["split"] == name].to_csv(f"data/ames_random_{name}.csv", index=False)

    # ChEMBL
    chembl = load_chembl()
    cleaned_chembl = cleaning_steps(chembl, "smiles", batch_size, passi=True)
    for name in ["train", "valid", "test"]:
        cleaned_chembl[cleaned_chembl["split"] == name].to_csv(f"data/chembl_full_{name}.csv", index=False)

    unique_chembl = (
        pd.merge(
            cleaned_chembl, cleaned_ames[["canonical_smiles"]], on=["canonical_smiles"], how="left", indicator=True
        )
        .query('_merge == "left_only"')
        .drop(columns=["_merge"])
        .dropna()
    )

    for name in ["train", "valid", "test"]:
        unique_chembl[unique_chembl["split"] == name].to_csv(f"data/chembl_unique_{name}.csv", index=False)


if __name__ == "__main__":
    main()
