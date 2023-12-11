import pandas as pd
from tdc.generation import MolGen
from tdc.single_pred import Tox


def import_ames() -> pd.DataFrame:
    data = Tox(name="AMES")

    scaffold_dfs = data.get_split(method="scaffold")
    scaffold = pd.concat({k: scaffold_dfs[k] for k in ["test", "valid", "train"]}, axis=0)
    scaffold = scaffold.reset_index(level=0, drop=False).rename(columns={"level_0": "scaffold"}).reset_index(drop=True)

    random_dfs = data.get_split(method="random")
    random = pd.concat({k: random_dfs[k] for k in ["test", "valid", "train"]}, axis=0)
    random = random.reset_index(level=0, drop=False).rename(columns={"level_0": "random"}).reset_index(drop=True)

    df = scaffold.merge(random[["Drug_ID", "random"]], on="Drug_ID", how="left")
    return df


def import_chembl() -> pd.DataFrame:
    data = MolGen(name="ChEMBL_V29")
    dfs = data.get_split(method="random")
    df = pd.concat({k: dfs[k] for k in ["test", "valid", "train"]}, axis=0)
    df = df.reset_index(level=0, drop=False).rename(columns={"level_0": "random"}).reset_index(drop=True)
    return df
