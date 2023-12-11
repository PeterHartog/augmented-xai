from typing import Any, Optional, Union

from rdkit import Chem


class Converter:
    @staticmethod
    def smile2mol(smile: str) -> Optional[Any]:
        try:
            return Chem.MolFromSmiles(smile)
        except Exception:
            return None

    @staticmethod
    def smile2reaction(smile: str) -> Optional[Any]:
        try:
            return Chem.rdChemReactions.ReactionFromSmarts(smile, useSmiles=True)
        except Exception:
            return None

    @staticmethod
    def mol2canonical(mol: Optional[Any]) -> Union[str, None]:
        try:
            return Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                kekuleSmiles=False,
                canonical=True,
                allBondsExplicit=False,
                allHsExplicit=False,
            )
        except Exception:
            return None

    @staticmethod
    def smile2canonical(smile: str) -> Union[str, None]:
        try:
            return Chem.MolToSmiles(
                Chem.MolFromSmiles(smile),
                isomericSmiles=True,
                kekuleSmiles=False,
                canonical=True,
                allBondsExplicit=False,
                allHsExplicit=False,
            )
        except Exception:
            return None


def heavy_atoms(smile: str) -> int:
    return Chem.GetNumHeavyAtoms(Chem.MolFromSmiles(smile))
