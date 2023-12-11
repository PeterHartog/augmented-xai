# beam search orginially written by Paula Torren Peraire. Adapted for this project.

from typing import Any, Optional

import torch
from aidd_codebase.registries import AIDD
from torch import Tensor

# TODO: test for batch_first = False


class Sampler:
    def __init__(
        self,
        decoder: Any,
        embed: Optional[Any] = None,
        fc_out: Optional[Any] = None,
        max_seq_len: int = 112,
        batch_first: bool = True,
        device: str = "cuda:0",
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        k: Optional[int] = None,
    ) -> None:
        assert max_seq_len is not None, "max_seq_len needs to be defined"

        self.embed = embed
        self.decoder = decoder
        self.fc_out = fc_out
        self.device = device
        self.k = k

        self.batch_first = batch_first
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len

        self.soft_max = torch.nn.Softmax(dim=-1)

    def generate_starting_tokens(self, batch_size: int, max_seq_len: int, k: Optional[int] = None) -> Tensor:
        """Generate a tensor with tokens, each starting with a BOS token and filled with PAD tokens.

        Parameters
        ----------
        batch_size : int
            size of the batch
        max_seq_len : int
            maximum length of the sequences
        k : Optional[int], optional
            additional dimention for beam search, by default None

        Returns
        -------
        Tensor
            starting tokens: (B, S (, k)) | (S, B (, k))
        """
        shape = [batch_size, max_seq_len] if self.batch_first else [max_seq_len, batch_size]
        starting_tokens = torch.full(shape, self.pad_idx, device=self.device)
        if self.batch_first:
            starting_tokens[:, 0] = self.bos_idx
        else:
            starting_tokens[0, :] = self.bos_idx
        if k:
            starting_tokens = starting_tokens.unsqueeze(-1).expand(shape + [k])
        return starting_tokens

    def create_finish_mask(self, seq: Tensor, i: int) -> Tensor:
        seq = seq[:, :i] if self.batch_first else seq[:i, :]
        mask = torch.where((seq == self.pad_idx) | (seq == self.eos_idx), True, False)
        mask = ~torch.any(mask, dim=1 if self.batch_first else 0).to(seq.device)
        return mask

    def index_step(
        self,
        i: int,
        eos_mask: Tensor,
        tgt: Tensor,
        memory: Tensor,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """Selects the indexes corresponding to unfinished sequences and up to index i from the tensors.

        Parameters
        ----------
        i : int
            index up to which to select tokens
        eos_mask : Tensor
            mask of sequences unfinished: (B)
        tgt : Tensor
            tokens from which to predict: (B, S)
        memory : Tensor
            memory tensor: (B, S)
        memory_padding_mask : Optional[Tensor], optional
            padding mask for tensor: (B, S), by default None

        Returns
        -------
        tuple[Tensor, Tensor, Optional[Tensor]]
            tgt, memory and memory padding mask: (B-eos_mask, S[:i]), (B-eos_mask, S), (B-eos_mask, S)
        """
        step_tokens = tgt[eos_mask, :i] if self.batch_first else tgt[:i, eos_mask]
        step_memory = memory[eos_mask, :] if self.batch_first else memory[:, eos_mask]
        if memory_padding_mask is not None:
            step_mem_padding_mask = (
                memory_padding_mask[eos_mask, :] if self.batch_first else memory_padding_mask[:, eos_mask]
            )
        else:
            step_mem_padding_mask = None
        return step_tokens, step_memory, step_mem_padding_mask

    def _step(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ):
        x = self.embed(tgt) if self.embed is not None else tgt
        x = self.decoder(x, memory, tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask)
        logits = self.fc_out(x) if self.fc_out is not None else x
        output = self.soft_max(logits)
        output = torch.log10(output).to(self.device)
        return output


@AIDD.ModelRegistry.register(  # type: ignore
    key="greedy_search", author="Paula Torren Peraire, Fabian Kruger", credit_type="acknowledgement"
)
class GreedySearch(Sampler):
    def search(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = memory.shape[0 if self.batch_first else 1]
        greedy_tokens = self.generate_starting_tokens(batch_size, self.max_seq_len)

        for i in range(1, self.max_seq_len - 1):
            eos_mask = self.create_finish_mask(greedy_tokens, i)
            if torch.all(~eos_mask):  # end if all sequences are already at their end
                break

            step_tokens, step_memory, step_mem_padding_mask = self.index_step(
                i, eos_mask, greedy_tokens, memory, memory_padding_mask
            )

            ll = self._step(
                tgt=step_tokens,
                memory=step_memory,
                tgt_mask=None,  # Not needed: single step prediction
                memory_mask=memory_mask,
                tgt_padding_mask=None,  # Not needed: already filtered out
                memory_padding_mask=step_mem_padding_mask,
            )
            ll = ll[:, -1, :] if self.batch_first else ll[-1, :, :]  # (B-eos, V)
            if self.batch_first:
                greedy_tokens[eos_mask, i] = torch.argmax(ll, dim=-1)
            else:
                greedy_tokens[i, eos_mask] = torch.argmax(ll, dim=-1)
        return greedy_tokens[:, 1:] if self.batch_first else greedy_tokens[1:, :]


@AIDD.ModelRegistry.register(
    key="beam_search", author="Paula Torren Peraire, Fabian Kruger", credit_type="acknowledgement"
)
class BeamSearch(Sampler):
    def __init__(
        self,
        decoder: Any,
        embed: Any | None = None,
        fc_out: Any | None = None,
        max_seq_len: int = 112,
        batch_first: bool = True,
        device: str = "cuda:0",
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        k: int | None = None,
    ) -> None:
        super().__init__(decoder, embed, fc_out, max_seq_len, batch_first, device, pad_idx, bos_idx, eos_idx, k)
        assert k is not None, "k has to be defined when using beamsearch"
        self.k: int = k

    def update_beams(self, beam_tokens: Tensor, score: Tensor, idx: Tensor, i: int) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = beam_tokens.shape
        score, idx = score.transpose(1, 2).reshape(batch_size, -1), idx.transpose(1, 2).reshape(batch_size, -1)
        beam_tokens = (
            beam_tokens.unsqueeze(-1)
            .expand((batch_size, seq_len, self.k, self.k))
            .transpose(2, 3)
            .reshape(batch_size, seq_len, -1)
        )

        score, k_idx = torch.topk(score, self.k)
        idx = torch.gather(idx, 1, k_idx)  # (B, k)
        beam_tokens = torch.gather(beam_tokens, 0, k_idx.unsqueeze(1).repeat(1, seq_len, 1))  # (B, S, k)

        beam_tokens[:, i, :] = idx

        return beam_tokens.to(self.device), score.to(self.device)

    def _beam_step(
        self,
        i: int,
        max_score: Tensor,
        beam_tokens: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = beam_tokens.shape[0 if self.batch_first else 1]

        idx = torch.full((batch_size, self.k, self.k), self.pad_idx, dtype=torch.int64, device=self.device)  # b, k, k
        score = max_score.unsqueeze(-1).expand(max_score.shape + (self.k,)).clone()  # b, k, k

        for k_ in range(self.k):
            k_tokens = beam_tokens[:, :, k_]
            eos_mask = self.create_finish_mask(k_tokens, i)
            if torch.all(~eos_mask):
                continue

            step_tokens, step_memory, step_mem_padding_mask = self.index_step(
                i, eos_mask, k_tokens, memory, memory_padding_mask
            )
            ll = self._step(
                tgt=step_tokens,
                memory=step_memory,
                tgt_mask=None,  # Not needed: single step prediction
                memory_mask=memory_mask,
                tgt_padding_mask=None,  # Not needed: already filtered out
                memory_padding_mask=step_mem_padding_mask,
            )

            ll = ll[:, -1, :] if self.batch_first else ll[-1, :, :]  # (B-eos, V)
            k_score, k_idx = torch.topk(ll, self.k)

            score[~eos_mask, 1:, k_] = -float("inf")
            score[eos_mask, :, k_] = score[eos_mask, :, k_] + k_score
            idx[eos_mask, :, k_] = k_idx

            if i == 1:
                score[:, :, 1:] = -float("inf")
                break  # Initialize beams before diverging

        beam_tokens, score = self.update_beams(beam_tokens, score, idx, i)
        return beam_tokens, score

    def search(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = memory.shape[0 if self.batch_first else 1]
        beam_tokens = self.generate_starting_tokens(batch_size, self.max_seq_len, self.k)
        score = torch.zeros([batch_size, self.k]).to(self.device)
        for i in range(1, self.max_seq_len):
            beam_tokens, score = self._beam_step(i, score, beam_tokens, memory, memory_mask, memory_padding_mask)
            eos_mask = self.create_finish_mask(beam_tokens, i)
            if torch.all(~eos_mask):
                break
        return beam_tokens[:, 1:, :] if self.batch_first else beam_tokens[1:, :, :]
