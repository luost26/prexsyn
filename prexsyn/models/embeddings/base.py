import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class Embedding:
    embedding: torch.Tensor
    padding_mask: torch.Tensor

    def __getitem__(self, s: slice) -> "Embedding":
        return Embedding(self.embedding[s], self.padding_mask[s])

    def pad(self, target_length: int) -> "Embedding":
        if self.sequence_length >= target_length:
            return self
        pad_length = target_length - self.sequence_length
        padded_embedding = torch.nn.functional.pad(self.embedding, (0, 0, 0, pad_length), value=0.0)
        padded_padding_mask = torch.nn.functional.pad(self.padding_mask, (0, pad_length), value=float("-inf"))
        return Embedding(padded_embedding, padded_padding_mask)

    def join(self, *others: "Embedding") -> "Embedding":
        if not others:
            return self
        return Embedding(
            torch.cat([self.embedding] + [o.embedding for o in others], dim=-2),
            torch.cat([self.padding_mask] + [o.padding_mask for o in others], dim=-1),
        )

    @property
    def device(self) -> torch.device:
        return self.embedding.device

    @property
    def batch_size(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def dim(self) -> int:
        return int(self.embedding.shape[2])

    def expand(self, n: int) -> "Embedding":
        return Embedding(
            embedding=self.embedding.expand(n, -1, -1),
            padding_mask=self.padding_mask.expand(n, -1),
        )

    def repeat(self, n: int) -> "Embedding":
        return Embedding(
            embedding=self.embedding.repeat(n, 1, 1),
            padding_mask=self.padding_mask.repeat(n, 1),
        )

    def repeat_interleave(self, n: int) -> "Embedding":
        return Embedding(
            embedding=self.embedding.repeat_interleave(n, dim=0),
            padding_mask=self.padding_mask.repeat_interleave(n, dim=0),
        )

    @staticmethod
    def create_full_padding_mask(h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.shape[:-1], dtype=h.dtype, device=h.device)

    @staticmethod
    def concat_along_seq_dim(*embeddings: "Embedding") -> "Embedding":
        h = torch.cat([e.embedding for e in embeddings], dim=1)
        m = torch.cat([e.padding_mask for e in embeddings], dim=1)
        return Embedding(h, m)

    @staticmethod
    def concat_along_batch_dim(*embeddings: "Embedding", max_length: int | None = None) -> "Embedding":
        if max_length is None:
            max_length = max(e.sequence_length for e in embeddings)
        padded_embeddings = [e.pad(max_length) for e in embeddings]
        h = torch.cat([e.embedding for e in padded_embeddings], dim=0)
        m = torch.cat([e.padding_mask for e in padded_embeddings], dim=0)
        return Embedding(h, m)
