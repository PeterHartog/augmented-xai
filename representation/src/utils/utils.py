from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def save_model_to_onnx(model, model_path, dummy_input):
    torch.onnx.export(model, dummy_input, model_path)


# def generate_square_subsequent_mask(sz: int) -> Tensor:
#     """Generates an upper-trianghular matrix of -inf with zeros on diag."""
#     return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-trianghular matrix of -inf with zeros on diag."""
    # return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)  # mine
    return ~(torch.tril(torch.ones(sz, sz)) == 1)  # Fabian
    # return torch.from_numpy(np.triu(np.ones((sz, sz)), k=1).astype("uint8") == 0)  # Allesandro


def create_src_mask(src: Tensor, batch_first: bool = False) -> Tensor:
    dim = 1 if batch_first else 0
    return torch.zeros((src.shape[dim], src.shape[dim])).type(torch.bool)


def create_tgt_mask(tgt: Tensor, batch_first: bool = False) -> Tensor:
    dim = 1 if batch_first else 0
    return generate_square_subsequent_mask(tgt.shape[dim])


def create_padding_mask(seq: Tensor, pad_idx: int) -> Tensor:
    return seq == pad_idx


def pad_sequences(
    sequences: list[Tensor],
    batch_first: bool = True,
    padding_value: int = 0,
    max_len: Optional[int] = None,
) -> Tensor:
    B, L = (
        len(sequences),
        max(seq.size(0) for seq in sequences) if not max_len else max_len,
    )
    out_dims = (B, L, *sequences[0].size()[1:]) if batch_first else (L, B, *sequences[0].size()[1:])
    out_tensor = torch.zeros(out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, : tensor.shape[0], ...] = tensor
        else:
            out_tensor[: tensor.shape[0], i, ...] = tensor

    return out_tensor


def mask_sequences(
    src: Tensor,
    pad_idx: int,
    msk_idx: int,
    vocab_size: int,
    mask_p: float = 0.15,
    rand_p: float = 0.1,
    unchanged_p: float = 0.1,
) -> Tensor:
    pad_mask = (src != pad_idx).bool()
    num_mask = int(src[pad_mask].size(0) * mask_p)
    if num_mask == 0:
        num_mask = 1

    # Choose 15% of the token positions at random for prediction
    pos_idx = torch.randperm(src[pad_mask].size(0))[:num_mask]
    num_random, num_unchanged = int(num_mask * rand_p), int(num_mask * unchanged_p)
    random_idx = pos_idx[:num_random]
    unchanged_idx = pos_idx[num_random:num_unchanged]

    # Replace the chosen token positions in the src with [MASK] token 80% of the time
    # Replace with the unchanged i-th token 10% of the time
    # Replace with a random token 10% of the time
    src_masked = src.clone()
    src_masked[pos_idx] = msk_idx
    src_masked[random_idx] = torch.randint(0, vocab_size, (len(random_idx),)).float()
    src_masked[unchanged_idx] = src[unchanged_idx]

    # # Create the attention mask
    # attn_mask = torch.zeros_like(src_masked)
    # attn_mask[pos_idx] = 1
    return src_masked  # , attn_mask, ~pad_mask


def df_to_torch(df: pd.DataFrame) -> Union[torch.Tensor, np.ndarray]:
    try:
        return torch.from_numpy(df.to_numpy().astype(np.float32))
    except ValueError:
        return df.to_numpy()
