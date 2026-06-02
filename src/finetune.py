# finetune.py -- ModelScope download + HuggingFace fine-tuning (full / LoRA)

import os
import json
import shutil
import gc
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from modelscope import snapshot_download
from peft import LoraConfig, get_peft_model, TaskType

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "models")


def download_model(model_id: str) -> str:
    """Download a model from ModelScope and return the local path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_dir = snapshot_download(model_id, cache_dir=CACHE_DIR)
    return model_dir


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
):
    """
    Generator-based fine-tuning with streaming progress updates.
    Yields status strings during progress, and final model path at the end.
    """
    if output_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, "output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Download model ----
    yield "Downloading model from ModelScope..."
    try:
        model_dir = download_model(model_id)
        yield f"Model downloaded to: {model_dir}"
    except Exception as e:
        raise RuntimeError(f"Failed to download model '{model_id}': {e}")

    # ---- Load model & tokenizer ----
    yield "Loading model and tokenizer..."
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    # ---- Apply LoRA ----
    if use_lora:
        yield f"Applying LoRA (r={lora_r}, alpha={lora_alpha})..."
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        yield f"LoRA ready: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.1f}%)"
    else:
        yield "Full fine-tuning mode (no LoRA)..."

    # ---- Load dataset ----
    yield "Processing dataset..."
    clean_json = json_path.strip().strip("\"'")
    with open(clean_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        # Auto-detect field names
        user_key = "instruction" if "instruction" in item else list(item.keys())[0]
        assistant_key = "output" if "output" in item else list(item.keys())[-1]
        messages = [
            {"role": "user", "content": str(item[user_key])},
            {"role": "assistant", "content": str(item[assistant_key])},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    hf_dataset = HFDataset.from_dict({"text": texts})
    tokenized_dataset = hf_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05)
    yield f"Dataset ready: {len(tokenized_dataset['train'])} train / {len(tokenized_dataset['test'])} validation samples"

    model.train()

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, ".tmp_finetune"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=(device.type == "cuda"),
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    yield "Starting training..."
    trainer.train()

    # ---- Save ----
    save_path = os.path.join(output_dir, save_name)
    os.makedirs(save_path, exist_ok=True)

    if use_lora:
        yield "Merging LoRA weights..."
        model = model.merge_and_unload()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Cleanup temp training dir
    tmp = os.path.join(output_dir, ".tmp_finetune")
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)

    # Delete downloaded model from cache to free disk space
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)

    # Free GPU memory
    del model
    del trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    yield f"SAVED:{save_path}"