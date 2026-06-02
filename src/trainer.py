# trainer.py --- training loop with streaming support

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import ModelConfig
from .model import GPT
from .tokenizer import CharTokenizer


def collate_batch(batch: list, pad_token_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_padded = torch.full((len(xs), max_len), pad_token_id, dtype=torch.long)
    y_padded = torch.full((len(ys), max_len), pad_token_id, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        x_padded[i, :x.size(0)] = x
        y_padded[i, :y.size(0)] = y
    return x_padded, y_padded


def train_model_stream(
    dataset,
    tokenizer: CharTokenizer,
    epochs: int = 10,
    learning_rate: float = 3e-4,
    batch_size: int = 16,
    model_size: str = "small",
    pretrained_embed_weight: torch.Tensor | None = None,
):
    """
    流式训练生成器：每完成一个 epoch 就 yield 进度信息。
    训练完成后 yield 模型对象。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size_map = {
        "small": ModelConfig.small(),
        "medium": ModelConfig.medium(),
        "large": ModelConfig.large(),
        "xlarge": ModelConfig.xlarge(),
    }
    config = size_map.get(model_size, ModelConfig.small())
    config.vocab_size = tokenizer.vocab_size

    # Auto-detect longest sequence in dataset and adjust block_size
    max_seq_len = 0
    for i in range(len(dataset)):
        x, y = dataset[i]
        max_seq_len = max(max_seq_len, x.size(0) + 1)
    max_seq_len = max(max_seq_len, 16)
    max_seq_len = min(max_seq_len, 2048)
    if max_seq_len > config.block_size:
        config.block_size = max_seq_len

    # 如果传入了预训练 embedding，确保 device 一致
    if pretrained_embed_weight is not None:
        pretrained_embed_weight = pretrained_embed_weight.to(device)

    model = GPT(config, pretrained_embed_weight=pretrained_embed_weight).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, config.pad_token_id),
        drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.1)

    # ---- 训练配置信息 ----
    embed_src = f"(pretrained: {pretrained_embed_weight is not None})"
    yield (
        f"## 训练配置\n\n"
        f"- 设备: **{device}**\n"
        f"- 模型规模: **{model_size}** "
        f"(参数量: {total_params:,} | "
        f"词表: {tokenizer.vocab_size} | "
        f"block_size: {config.block_size} (dataset max: {max_seq_len})\n"
        f"- 训练轮数: **{epochs}** | 学习率: **{learning_rate:.1e}** | "
        f"批次大小: **{batch_size}**\n"
        f"- 样本数: **{len(dataset)}** | 每轮步数: **{len(dataloader)}**\n"
        f"- Embedding 初始化: **{embed_src}**\n"
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        step_count = 0

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                step_count += 1

        if step_count > 0:
            scheduler.step()
        avg_loss = total_loss / max(step_count, 1)
        elapsed = time.time() - epoch_start
        lr_now = scheduler.get_last_lr()[0]

        yield (
            f"### Epoch {epoch}/{epochs}\n\n"
            f"| 指标 | 值 |\n|------|----|\n"
            f"| Loss | {avg_loss:.4f} |\n"
            f"| 学习率 | {lr_now:.2e} |\n"
            f"| 用时 | {elapsed:.1f}s |"
        )

    # 训练完成，返回模型
    yield model


def train_model(
    dataset,
    tokenizer: CharTokenizer,
    epochs: int = 10,
    learning_rate: float = 3e-4,
    batch_size: int = 16,
    model_size: str = "small",
    progress_callback=None,
) -> GPT:
    """同步训练（保留旧接口兼容性）"""
    gen = train_model_stream(
        dataset=dataset,
        tokenizer=tokenizer,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_size=model_size,
    )
    result = None
    for item in gen:
        if isinstance(item, GPT):
            result = item
        elif progress_callback and isinstance(item, str):
            progress_callback(item)
    if result is None:
        raise RuntimeError("训练未产生模型")
    return result


def save_model(model: GPT, tokenizer: CharTokenizer, output_dir: str, model_name: str):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pth")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "dropout": model.config.dropout,
        },
        "tokenizer": tokenizer.to_dict(),
        "model_name": model_name,
    }
    torch.save(checkpoint, model_path)
    return model_path
