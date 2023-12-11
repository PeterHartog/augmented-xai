from abc import ABC, abstractmethod

from torch import Tensor


class AbstractTokenizer(ABC):
    pad_idx: int
    bos_idx: int
    eos_idx: int
    unk_idx: int
    msk_idx: int
    vocab_size: int
    max_seq_len: int

    @abstractmethod
    def tokenize(self, smile: str) -> list[int]:
        """Generates integers for to the tokens for a smiles string."""

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Generates smiles string from tokens."""

    @abstractmethod
    def sequentialize(self, tokens: list[int]) -> list:
        """Adds BOS, EOS and padding to tokens."""

    @abstractmethod
    def desequentialize(self, tokens: list[int]) -> list:
        """Removes BOS, EOS and padding from tokens."""

    @abstractmethod
    def tensorize(self, tokens: list[int]) -> Tensor:
        """Create tensor from the tokens."""

    @abstractmethod
    def detensorize(self, tensor: Tensor) -> list[int]:
        """Transforms a tensor back into a list of tokens."""

    def smile_return(self, tensor: Tensor) -> str:
        return self.detokenize(self.detensorize(tensor))

    def smile_prep(self, smile: str) -> Tensor:
        return self.tensorize(self.tokenize(smile))
