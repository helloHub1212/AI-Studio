# config.py — Model & Training Configuration

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    use_swiglu: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    @classmethod
    def small(cls) -> 'ModelConfig':
        return cls(n_layer=4, n_head=4, n_embd=128, block_size=128)

    @classmethod
    def medium(cls) -> 'ModelConfig':
        return cls(n_layer=6, n_head=8, n_embd=256, block_size=256)

    @classmethod
    def large(cls) -> 'ModelConfig':
        return cls(n_layer=8, n_head=8, n_embd=512, block_size=512)

    @classmethod
    def xlarge(cls) -> 'ModelConfig':
        return cls(n_layer=12, n_head=12, n_embd=768, block_size=768)

    @classmethod
    def max(cls) -> 'ModelConfig':
        return cls(n_layer=24, n_head=16, n_embd=1024, block_size=1024)


SIZE_PRESETS = {
    "small": ModelConfig.small(),
    "medium": ModelConfig.medium(),
    "large": ModelConfig.large(),
    "xlarge": ModelConfig.xlarge(),
    "max": ModelConfig.max(),
}

SIZE_LABELS = {
    "small": "small (4L-128d)",
    "medium": "medium (6L-256d)",
    "large": "large (8L-512d)",
    "xlarge": "xlarge (12L-768d)",
    "max": "max (24L-1024d)",
}