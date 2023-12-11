from rdkit import Chem
from rdkit.Chem import AllChem


def fingerprint_converter(smi: str, radius: int = 2, use_features: bool = False, nbit: int = 1024):
    """
    Returns a Morgan fingerprint given a SMILES of a molecule.
    :param smi: SMILES string
    :param radius: Connection radius around which the fingerprint is drawn.
    :param use_features: Use features or
    :param nbit: Number of bits in the final bitvector.
    :return : A bitvector of nbit size.
    """
    mol = Chem.MolFromSmiles(smi)  # type: ignore
    bit_vec = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore
        mol,
        radius=radius,
        useFeatures=use_features,
        nBits=nbit,
    )
    return bit_vec
