from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_physchem(smi):
    mol = Chem.MolFromSmiles(smi)

    alogp = Descriptors.MolLogP(mol)
    min_charge = Descriptors.MinPartialCharge(mol)
    max_charge = Descriptors.MaxPartialCharge(mol)
    val_electrons = Descriptors.NumValenceElectrons(mol)
    hdb = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    balaban_j = Descriptors.BalabanJ(mol)
    refrac = Descriptors.MolMR(mol)
    tpsa = Descriptors.TPSA(mol)
    return [alogp, min_charge, max_charge, val_electrons, hdb, hba, balaban_j, refrac, tpsa]
