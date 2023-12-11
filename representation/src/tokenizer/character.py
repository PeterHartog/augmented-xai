from dataclasses import dataclass

import torch
from aidd_codebase.registries import AIDD
from torch import Tensor

from representation.src.tokenizer.abstract import AbstractTokenizer
from representation.src.utils.inspect_kwargs import set_kwargs


@AIDD.DataRegistry.register_arguments(key="character_tokenizer")
@dataclass(unsafe_hash=True)
class TokenArguments:
    # Define special symbols and indices
    pad_idx: int = 0  # Padding
    bos_idx: int = 1  # Beginning of Sequence
    eos_idx: int = 2  # End of Sequence
    unk_idx: int = 3  # Unknown Value
    msk_idx: int = 4  # Mask

    # Our vocabulary
    vocab: str = " ^$?_#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
    max_seq_len: int = 175


@AIDD.DataRegistry.register(key="character_tokenizer", author="Peter Hartog", credit_type="None")
class CharTokenizer(AbstractTokenizer):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        args = set_kwargs(TokenArguments, **kwargs)

        self.vocab_size = len(args.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(args.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(args.vocab)}

        self.pad_idx = args.pad_idx
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.msk_idx = args.msk_idx
        self.max_seq_len = args.max_seq_len
        self.max_tensor_len = self.max_seq_len + 2

    def tokenize(self, smile: str) -> list[int]:
        """Generates integers for to the tokens for a smiles string."""
        return [self.char_to_ix[ch] for ch in smile]

    def detokenize(self, tokens: list[int]) -> str:
        """Generates smiles string from tokens."""
        return "".join([self.ix_to_char[token] for token in tokens])

    def sequentialize(self, tokens: list[int]) -> list:
        """Adds BOS, EOS and padding to tokens."""
        return [self.bos_idx] + tokens + [self.eos_idx]

    def desequentialize(self, tokens: list[int]) -> list:
        """Removes BOS, EOS and padding from tokens."""
        return [token for token in tokens if token not in [self.bos_idx, self.eos_idx, self.pad_idx]]

    def tensorize(self, tokens: list[int]) -> Tensor:
        """Create tensor from the tokens."""
        return torch.tensor(self.sequentialize(tokens))

    def detensorize(self, tensor: Tensor) -> list[int]:
        """Transforms a tensor back into a list of tokens."""
        return self.desequentialize(tensor.tolist())
