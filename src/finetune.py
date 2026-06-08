# finetune.py — ModelScope Download + HuggingFace Fine-tuning (Full / LoRA / QLoRA)

import os
import json
import shutil
import gc
import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from modelscope import snapshot_download
from peft import LoraConfig, get_peft_model, TaskType

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "models")


def download_model(model_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return snapshot_download(model_id, cache_dir=CACHE_DIR)


def fine_tune_stream(
    model_id: str,
    json_path: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    save_name: str = "finetuned_model",
    output_dir: str = None,
    use_4bit: bool = False,
    use_checkpoint: bool = True,
    grad_accum_steps: int = 1,
    num_workers: int = 0,
    warmup_ratio: float = 0.05,
    eval_ratio: float = 0.1,
    max_seq_length: int = 512,
    packing: bool = False,
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.005,
):
    """
    Generator-based fine-tuning with streaming progress updates.
    Yields status strings, final model path at the end.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Download model ----
    yield "Downloading model from ModelScope..."
    try:
        model_dir = download_model(model_id)
        yield f"Model downloaded to: {model_dir}"
    except Exception as e:
        raise RuntimeError(f"Failed to download model '{model_id}': {e}")

    def _log_vram(tag: str):
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            line = f"  [{tag}] VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            print(line)
            yield line

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Load model ----
    yield "Loading model and tokenizer..."

    if use_4bit:
        yield "4-bit quantization enabled (QLoRA)..."
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        )
    else:
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=dtype,
        )

    # Enable Flash Attention 2 / SDPA if available
    if hasattr(model.config, "use_flash_attention"):
        model.config.use_flash_attention = True
    if hasattr(model.config, "attn_implementation"):
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for _ in _log_vram("after model load"):
        yield _

    if use_4bit and not use_lora:
        yield "4-bit requires LoRA, enabling automatically..."
        use_lora = True

    # ---- Apply LoRA ----
    if use_lora:
        yield f"Applying LoRA (r={lora_r}, alpha={lora_alpha})..."
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=0.1, target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        yield f"LoRA ready: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)"
        for _ in _log_vram("after LoRA"):
            yield _
    else:
        yield "Full fine-tuning mode..."

    # ---- Gradient checkpointing ----
    if use_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, "model"):
            model.model.gradient_checkpointing_enable()
        if use_lora:
            model.enable_input_require_grads()
            if hasattr(model, "model"):
                model.model.enable_input_require_grads()
    for _ in _log_vram("after checkpoint"):
        yield _

    # ---- Load dataset ----
    yield "Processing dataset..."
    clean_json = json_path.strip().strip('"').strip("'").strip()
    ext = os.path.splitext(clean_json)[1].lower()

    records = []
    if ext == ".jsonl":
        with open(clean_json, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(clean_json, "r", encoding="utf-8") as f:
            records = json.load(f)

    texts = []
    for item in records:
        if isinstance(item, dict) and "conversation" in item and isinstance(item["conversation"], list):
            system_prompt = item.get("system", "").strip()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for turn in item["conversation"]:
                human = str(turn.get("human", "")).strip()
                assistant = str(turn.get("assistant", "")).strip()
                if not human and not assistant:
                    continue
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        else:
            user_key = "instruction" if "instruction" in item else list(item.keys())[0]
            assistant_key = "output" if "output" in item else list(item.keys())[-1]
            messages = [
                {"role": "user", "content": str(item[user_key])},
                {"role": "assistant", "content": str(item[assistant_key])},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    hf_dataset = HFDataset.from_dict({"text": texts})

    if packing:
        def tokenize_and_pack(examples):
            tokenized = tokenizer(examples["text"], truncation=False, padding=False, return_attention_mask=False)
            concatenated = {k: sum(v, []) for k, v in tokenized.items()}
            total_length = len(concatenated["input_ids"])
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i:i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated.items()
            }
            return result

        tokenized_dataset = hf_dataset.map(
            tokenize_and_pack, batched=True, remove_columns=["text"],
            num_proc=min(4, os.cpu_count() or 1) if num_workers > 0 else 1,
        )
    else:
        tokenized_dataset = hf_dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=max_seq_length, padding=False),
            batched=True, remove_columns=["text"],
            num_proc=min(4, os.cpu_count() or 1) if num_workers > 0 else 1,
        )

    yield f"Dataset ready: {len(tokenized_dataset)} samples"

    # Train/val split
    val_dataset = None
    train_dataset = tokenized_dataset
    if eval_ratio > 0 and len(tokenized_dataset) > 1:
        val_size = max(1, int(len(tokenized_dataset) * eval_ratio))
        train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=val_size, seed=42).values()
        yield f"Train: {len(train_dataset)} | Val: {len(val_dataset)}"

    model.train()
    optim = "adamw_8bit" if (use_4bit or use_lora) else "adamw_torch_fused"

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, ".tmp_finetune"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch" if val_dataset else "no",
        eval_strategy="epoch" if val_dataset else "no",
        bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device.type == "cuda" and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=(not use_4bit and use_checkpoint),
        report_to="none",
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=device.type == "cuda",
        dataloader_persistent_workers=(num_workers > 0),
        remove_unused_columns=True,
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 早停回调（需有验证集 + 用户开启才生效）
    callbacks = []
    if val_dataset and early_stopping:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ))

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    for _ in _log_vram("before training start"):
        yield _

    from transformers import TrainerCallback
    class _VRAMLogCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated(0) / 1024**3
                r = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  [step {state.global_step} epoch {state.epoch:.2f}] VRAM={a:.2f}GB/{r:.2f}GB")
        def on_epoch_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated(0) / 1024**3
                r = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  [epoch end] VRAM={a:.2f}GB/{r:.2f}GB")

    trainer.add_callback(_VRAMLogCallback())
    yield "Starting training..."
    trainer.train()

    # ---- Save ----
    save_path = os.path.join(output_dir, save_name)
    os.makedirs(save_path, exist_ok=True)

    if use_lora:
        yield "Merging LoRA weights..."
        model = model.merge_and_unload()

    # Check actual parameter dtypes (lm_head is fp16 even on 4-bit → check all params)
    _std_dtypes = {torch.bfloat16, torch.float16, torch.float32}
    _all_std = all(p.dtype in _std_dtypes for p in model.parameters())

    if _all_std:
        # Ensure consistent save dtype (prevents accidental 4-bit save)
        target_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        model = model.to(target_dtype)
    else:
        _dtypes_found = set(p.dtype for p in model.parameters())
        yield (f"Note: model has non-standard dtypes {_dtypes_found}. "
               f"merge_and_unload may not have dequantized. Saving at original precision.")
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    tmp = os.path.join(output_dir, ".tmp_finetune")
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)

    del model, trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    yield f"SAVED:{save_path}"