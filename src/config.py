# config.py — 模型与训练配置

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 256       # 最大上下文长度
    n_layer: int = 6            # transformer 层数
    n_head: int = 8             # 注意力头数
    n_embd: int = 256           # 嵌入维度
    dropout: float = 0.1
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
