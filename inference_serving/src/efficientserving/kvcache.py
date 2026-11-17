"""
KV cache storage.

A cache holds the keys and values of every token a batch has processed so far, for
every layer, so each decoding step only has to run the model on the newest token.
"""
import torch


class KVCache:
    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers

    def sequence_length(self, layer_idx: int = 0) -> int:
        return 0 if self.keys[layer_idx] is None else self.keys[layer_idx].size(-2)

    def has_value(self, layer_idx: int) -> bool:
        return self.keys[layer_idx] is not None

    def load_kv(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.keys[layer_idx], self.values[layer_idx]

    def store_kv(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.keys[layer_idx] = key if self.keys[layer_idx] is None else torch.cat((self.keys[layer_idx], key), dim=-2)
        self.values[layer_idx] = value if self.values[layer_idx] is None else torch.cat((self.values[layer_idx], value), dim=-2)
        return self.keys[layer_idx], self.values[layer_idx]

    def release(self) -> None:
        for layer_idx in range(len(self.keys)):
            self.keys[layer_idx] = None
            self.values[layer_idx] = None


class KVCacheContinuous(KVCache):
    # TODO (KV cache for continuous batching): Implement!
    pass


class KVCachePaged(KVCache):
    # TODO (KV cache for paged continuous batching): Implement!
    pass
