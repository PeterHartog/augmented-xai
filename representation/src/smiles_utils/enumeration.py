import random
from typing import Optional

from rdkit import Chem

from representation.src.smiles_utils.randomization import get_randomize_smiles


def enumerate_smi(
    smi: str,
    n: int = 100,
    limit: Optional[int] = None,
    random_type: str = "unrestricted",
    keep_original: bool = True,
    random_seed: int = 1234,
) -> list[str]:
    random.seed(random_seed)
    mol = Chem.MolFromSmiles(smi)  # type: ignore
    smis = [get_randomize_smiles(mol, random_type=random_type) for _ in range(n)]
    smis = [smi] + smis if keep_original else smis
    smis = sorted(set(smis), key=smis.index)
    if limit is not None and len(smis) > limit:
        smis = smis[: (limit + 1 if keep_original else limit)]
    return smis
