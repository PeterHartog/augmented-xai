import random
from typing import Optional

from rdkit import Chem


def randomize_smiles(
    smi: str, random_type: str = "restricted", max_seq_len: Optional[int] = None, generation_limit: int = 100
):
    """
    from https://github.com/undeadpixel/reinvent-randomized/blob/master/utils/chem.py
    paper ref: https://doi.org/10.1186/s13321-019-0393-0
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: SMILES string
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smi)  # type: ignore
    if max_seq_len is not None:
        random_smi = None
        for _ in range(generation_limit):
            random_smi = get_randomize_smiles(mol, random_type=random_type)
            random_smi = random_smi if len(random_smi) <= max_seq_len else None
            if random_smi is not None:
                break
        return random_smi if random_smi is not None else smi
    else:
        return get_randomize_smiles(mol, random_type=random_type)


def get_randomize_smiles(mol, random_type: str = "restricted"):
    """
    from https://github.com/undeadpixel/reinvent-randomized/blob/master/utils/chem.py
    paper ref: https://doi.org/10.1186/s13321-019-0393-0
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: SMILES string
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)  # type: ignore
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)  # type: ignore
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)  # type: ignore
    raise ValueError("Type '{}' is not valid".format(random_type))
