import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mha_shape_check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    assert query.dim() == key.dim() == value.dim()
    return query.dim() == 3


def _reshape_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_batched: bool,
    batch_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rehsapes all tensors to be in dim 3 and transposes if batch_first."""
    if not is_batched:
        # unsqueeze if the input is unbatched
        query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)

    if batch_first:
        # make sure that the transpose op does not affect the "is" property
        if key is value:
            if query is key:
                query = key = value = query.transpose(1, 0)
            else:
                query, key = [x.transpose(1, 0) for x in (query, key)]
                value = key
        else:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
    return query, key, value


def _gather_shape_vars(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    correct_embed_dim: int,
    correct_num_heads: int,
    _qkv_same_embed_dim: bool,
) -> tuple[int, int, int, int, int]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == correct_embed_dim
    ), f"was expecting embedding dimension of {correct_embed_dim}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = int(embed_dim.div(correct_num_heads, rounding_mode="trunc"))
    else:
        head_dim = embed_dim // correct_num_heads

    assert (
        head_dim * correct_num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {correct_num_heads}"

    if not _qkv_same_embed_dim:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    return bsz, embed_dim, head_dim, src_len, tgt_len


def _prepare_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    is_batched: bool,
    is_causal: bool,
    need_weights: bool,
    dtype: torch.dtype,
    bsz: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if not is_batched:
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=dtype,
    )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    #
    # prep attention mask
    #
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    return attn_mask, key_padding_mask


def _in_projection(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    _qkv_same_embed_dim: bool,
    in_proj_weight: Optional[torch.Tensor],
    q_proj_weight: Optional[torch.Tensor],
    k_proj_weight: Optional[torch.Tensor],
    v_proj_weight: Optional[torch.Tensor],
    in_proj_bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute in-projection."""
    if _qkv_same_embed_dim:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        embed_dim = query.size(-1)
        # reshape to n, E and not E, n is deliberate for better memory coalescing and keeping same order as chunk()
        if query is value:
            if query is key:
                proj = F.linear(query, in_proj_weight, in_proj_bias)
                proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                q, k, v = proj[0], proj[1], proj[2]
            else:
                w_q, w_kv = in_proj_weight.split([embed_dim, embed_dim * 2])
                if in_proj_bias is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = in_proj_bias.split([embed_dim, embed_dim * 2])
                q_proj = F.linear(query, w_q, b_q)
                kv_proj = F.linear(key, w_kv, b_kv)
                kv_proj = kv_proj.unflatten(-1, (2, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                q, k, v = q_proj, kv_proj[0], kv_proj[1]
        else:
            w_q, w_k, w_v = in_proj_weight.chunk(3)
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = (
            F.linear(query, q_proj_weight, b_q),
            F.linear(key, k_proj_weight, b_k),
            F.linear(value, v_proj_weight, b_v),
        )
    return q, k, v


class CustomMultiheadAttention(nn.MultiheadAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_batched = _mha_shape_check(query, key, value)
        query, key, value = _reshape_tensors(query, key, value, is_batched, self.batch_first)

        bsz, embed_dim, head_dim, src_len, tgt_len = _gather_shape_vars(
            query, key, value, self.embed_dim, self.num_heads, self._qkv_same_embed_dim
        )

        # Calculate in-projection
        query, key, value = _in_projection(
            query,
            key,
            value,
            self._qkv_same_embed_dim,
            self.in_proj_weight,
            self.q_proj_weight,
            self.k_proj_weight,
            self.v_proj_weight,
            self.in_proj_bias,
        )

        attn_mask, key_padding_mask = _prepare_mask(
            attn_mask,
            key_padding_mask,
            is_batched,
            is_causal,
            need_weights,
            query.dtype,
            bsz,
            self.num_heads,
            tgt_len,
            src_len,
        )

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            key = torch.cat([key, self.bias_k.repeat(1, bsz, 1)])
            value = torch.cat([value, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        query = query.view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        key = key.view(key.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
        value = value.view(value.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, head_dim)
            key = torch.cat([key, torch.zeros(zero_attn_shape, dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value, torch.zeros(zero_attn_shape, dtype=value.dtype, device=value.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = key.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, tgt_len, -1)
                .reshape(bsz * self.num_heads, tgt_len, src_len)
            )

            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(bsz, -1, tgt_len, src_len)
                attn_mask = attn_mask_expanded + key_padding_mask
            elif attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, tgt_len, src_len).expand(bsz, self.num_heads, -1, -1)
                attn_mask = attn_mask_expanded + key_padding_mask
            else:
                raise ValueError("attn_mask is wrong dim")

        #
        # (deep breath) calculate attention and out projection
        #
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        query = query.view(bsz, self.num_heads, tgt_len, head_dim)
        key = key.view(bsz, self.num_heads, src_len, head_dim)
        value = value.view(bsz, self.num_heads, src_len, head_dim)
        dropout_p = 0.0 if not self.training else self.dropout  # adjust dropout probability

        #
        # scaled dot product
        #
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        attn_weight = attn_weight.view(bsz, self.num_heads, tgt_len, src_len)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_weight = attn_weight.squeeze(0)
            value = value.squeeze(0)

            attn_output = attn_weight @ value
            attn_output = attn_output.permute(1, 0, 2).contiguous().view(tgt_len, embed_dim)

            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, attn_output.size(1))
        else:
            attn_output = attn_weight @ value
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if self.batch_first:
                attn_output = attn_output.permute(1, 0, 2)

        if average_attn_weights:
            # optionally average attention weights over heads
            attn_weight = attn_weight.mean(dim=1)

        if not need_weights:
            attn_weight = None

        return attn_output, attn_weight
