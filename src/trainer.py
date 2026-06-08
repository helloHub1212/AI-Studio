# trainer.py — From-scratch GPT training loop with streaming progress
# Unified format with inference:
#   Training: <bos>{user_with_system}<eos><bos>assistant<eos>
#   Inference input: <bos>{user_with_system}<eos><bos>  (model generates until <eos>)

import os
import platform
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import ModelConfig, SIZE_PRESETS
from .model import GPT
from .tokenizer import CharTokenizer
from .dataset import make_collate_fn


def create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def train_model_stream(
    dataset,
    tokenizer: CharTokenizer,
    epochs: int = 10,
    learning_rate: float = 3e-4,
    batch_size: int = 16,
    model_size: str = "small",
    pretrained_embed_weight: torch.Tensor | None = None,
    use_checkpoint: bool = False,
    grad_accum_steps: int = 1,
    num_workers: int = 0,
    warmup_ratio: float = 0.05,
    eval_ratio: float = 0.1,
    compile_model: bool = True,
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.005,
):
    """Streaming training generator.

    Yields progress strings each epoch; the final yield is the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    config = SIZE_PRESETS.get(model_size, SIZE_PRESETS["small"])
    config.vocab_size = tokenizer.vocab_size

    # Adapt block_size to the longest sample in the dataset (cap at 2048)
    max_seq_len = 16
    for i in range(len(dataset)):
        x, _ = dataset[i]
        max_seq_len = max(max_seq_len, x.size(0) + 1)
    max_seq_len = min(max_seq_len, 2048)
    if max_seq_len > config.block_size:
        config.block_size = max_seq_len

    if model_size == "max":
        batch_size = min(batch_size, 4)
    elif model_size == "xlarge":
        batch_size = min(batch_size, 8)

    if pretrained_embed_weight is not None:
        pretrained_embed_weight = pretrained_embed_weight.to(device)

    model = GPT(config, pretrained_embed_weight=pretrained_embed_weight).to(device)
    if use_checkpoint:
        model.gradient_checkpointing = True
    total_params = sum(p.numel() for p in model.parameters())

    # torch.compile: needs Triton (Inductor backend) — only Linux supports it.
    # WSL2: default mode (cudagraphs can leak fds). Native Linux: reduce-overhead.
    # Windows / macOS: skip entirely (Triton unsupported → TritonMissing error).
    _uname = platform.uname()
    _is_wsl = "microsoft" in _uname.release.lower() or "wsl" in _uname.release.lower()
    _compile_supported = _uname.system == "Linux"  # only Linux has working Triton
    if compile_model and _compile_supported and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            _compile_mode = "default" if _is_wsl else "reduce-overhead"
            model = torch.compile(model, mode=_compile_mode, fullgraph=False)
        except Exception:
            pass

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            train_dtype = torch.bfloat16
            amp_dtype = torch.bfloat16
            use_scaler = False
            dtype_label = "bf16"
        else:
            train_dtype = torch.float16
            amp_dtype = torch.float16
            use_scaler = True
            dtype_label = "fp16"
    else:
        train_dtype = torch.float32
        amp_dtype = None
        use_scaler = False
        dtype_label = "fp32 (CPU)"
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Train/val split
    val_dataset = None
    train_dataset = dataset
    if eval_ratio > 0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * eval_ratio))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    # DataLoader
    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": device.type == "cuda" and not _is_wsl,
        "collate_fn": make_collate_fn(config.pad_token_id),
    }
    if num_workers > 0:
        nw = min(num_workers, 4)
        if _is_wsl:
            nw = min(nw, 2)  # WSL2 forkserver is fd-hungry
            dl_kwargs["multiprocessing_context"] = "spawn"  # avoid forkserver fd overhead
        dl_kwargs.update({
            "num_workers": nw,
            "persistent_workers": False,
        })

    train_loader = DataLoader(train_dataset, **dl_kwargs)
    val_loader = DataLoader(val_dataset, **dl_kwargs) if val_dataset else None

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, fused=(device.type == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)

    if early_stopping and val_dataset:
        es_status = f"开启 (patience={early_stopping_patience}, threshold={early_stopping_threshold})"
    elif val_dataset:
        es_status = f"关闭（开启需 patience={early_stopping_patience}）"
    else:
        es_status = "关闭（无验证集）"
    if not compile_model:
        compile_status = "关闭（用户设置）"
    elif not _compile_supported:
        compile_status = f"关闭（{_uname.system} 不支持 Triton）"
    elif device.type != "cuda":
        compile_status = "关闭（无 CUDA）"
    else:
        compile_status = f"开启（mode={'default (WSL2)' if _is_wsl else 'reduce-overhead'}）"
    yield (
        f"## 训练配置\n\n"
        f"- 设备: **{device}**\n"
        f"- 模型规模: **{model_size}** (参数量: {total_params:,} | 词表: {tokenizer.vocab_size} | block_size: {config.block_size})\n"
        f"- 训练轮数: **{epochs}** | 学习率: **{learning_rate:.1e}** | 批次: **{batch_size}** | 梯度累积: **{grad_accum_steps}** (有效: {batch_size * grad_accum_steps})\n"
        f"- 样本数: **{len(train_dataset)}** | 验证集: **{len(val_dataset) if val_dataset else 0}** | 每轮步数: **{steps_per_epoch}**\n"
        f"- 训练精度: **{dtype_label}** | DataLoader workers: **{num_workers}**\n"
        f"- Warmup steps: **{warmup_steps}** ({warmup_ratio*100:.0f}%) | 编译模式: **{compile_status}**\n"
        f"- 早停: **{es_status}**\n"
    )

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=train_dtype, enabled=use_amp):
                _, loss = model(x, targets=y)
                if loss is not None:
                    loss = loss / grad_accum_steps

            if loss is not None:
                scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if loss is not None:
                total_loss += loss.item() * grad_accum_steps
                step_count += 1

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.inference_mode():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, dtype=train_dtype, enabled=use_amp):
                        _, loss = model(x, targets=y)
                    if loss is not None:
                        val_total += loss.item()
                        val_count += 1
            val_loss = val_total / max(val_count, 1)
            model.train()

            if val_loss < best_val_loss - early_stopping_threshold:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

        avg_loss = total_loss / max(step_count, 1) if step_count > 0 else float('nan')
        elapsed = time.time() - epoch_start
        lr_now = scheduler.get_last_lr()[0]

        if (early_stopping and val_loader is not None
                and bad_epochs >= early_stopping_patience):
            yield (
                f"### Epoch {epoch}/{epochs}\n\n"
                f"| 指标 | 值 |\n|------|----|\n"
                f"| Train Loss | {avg_loss:.4f} |\n"
                f"| Val Loss | {val_loss:.4f} |\n"
                f"| 学习率 | {lr_now:.2e} |\n"
                f"| 用时 | {elapsed:.1f}s |\n"
                f"| Tokens/s | {step_count * batch_size * config.block_size / max(elapsed, 1e-6):.0f} |\n"
                f"\n**⏹ 早停触发**：Val Loss 连续 {early_stopping_patience} 轮未改善（最佳: {best_val_loss:.4f}）"
            )
            break

        yield (
            f"### Epoch {epoch}/{epochs}\n\n"
            f"| 指标 | 值 |\n|------|----|\n"
            f"| Train Loss | {avg_loss:.4f} |\n"
            + (f"| Val Loss | {val_loss:.4f} |\n" if val_loss is not None else "")
            + f"| 学习率 | {lr_now:.2e} |\n"
            f"| 用时 | {elapsed:.1f}s |\n"
            f"| Tokens/s | {step_count * batch_size * config.block_size / max(elapsed, 1e-6):.0f} |"
        )

    yield model


def save_model(model: GPT, tokenizer: CharTokenizer, output_dir: str, model_name: str,
               system_prompt: str | None = None):
    """Save a trained model and tokenizer. Persists system_prompt for inference."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pth")

    # Get underlying module (unwrap torch.compile wrapper if any)
    target = model
    if hasattr(model, "_orig_mod"):
        target = model._orig_mod

    checkpoint = {
        "model_state_dict": target.state_dict(),
        "config": {
            "vocab_size": target.config.vocab_size,
            "block_size": target.config.block_size,
            "n_layer": target.config.n_layer,
            "n_head": target.config.n_head,
            "n_embd": target.config.n_embd,
            "dropout": target.config.dropout,
        },
        "tokenizer": tokenizer.to_dict(),
        "model_name": model_name,
    }
    if system_prompt:
        checkpoint["system_prompt"] = system_prompt
    torch.save(checkpoint, model_path)
    return model_path
