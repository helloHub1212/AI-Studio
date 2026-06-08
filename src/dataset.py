# dataset.py — JSON/JSONL dataset loading + character-level conversation dataset
# Unified: training and inference share the same token format
#   <bos>{user_text_with_system}<eos><bos>{assistant_text}<eos>

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any


class ConversationDataset(Dataset):
    """Character-level conversation dataset.

    Each sample encodes: <bos>user_text<eos><bos>assistant_text<eos>
    Targets are the input shifted by one (next-token prediction).
    """

    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.pad_token_id
        self.samples: List[torch.Tensor] = []

        for user_input, assistant_output in pairs:
            user_ids = tokenizer.encode(user_input, add_special_tokens=True)
            assistant_ids = tokenizer.encode(assistant_output, add_special_tokens=True)
            all_ids = user_ids + assistant_ids
            if len(all_ids) > block_size:
                all_ids = all_ids[:block_size]
            if len(all_ids) >= 2:
                self.samples.append(torch.tensor(all_ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def detect_dataset_format(records: List[Dict]) -> Tuple[str, str]:
    if not records:
        raise ValueError('JSON dataset is empty')

    first = records[0]
    keys = list(first.keys())

    if len(keys) < 2:
        raise ValueError(f'Each record needs at least 2 fields, got {len(keys)}: {keys}')

    input_candidates = [
        'instruction', 'input', 'prompt', 'question', 'user', 'query',
        'text', 'context', 'src', 'source', 'human', 'request',
    ]
    output_candidates = [
        'output', 'response', 'answer', 'assistant', 'completion',
        'target', 'tgt', 'reply', 'result', 'gpt', 'response_text', 'value',
    ]

    input_key = None
    output_key = None
    for ik in input_candidates:
        if ik in keys:
            input_key = ik
            break
    for ok in output_candidates:
        if ok in keys and ok != input_key:
            output_key = ok
            break

    if input_key is None or output_key is None:
        # Fallback: use first/last available key
        remaining = [k for k in keys if k not in (input_key, output_key)]
        if input_key is None:
            input_key = remaining[0] if remaining else keys[0]
        if output_key is None:
            output_key = remaining[-1] if remaining else keys[-1]
        if input_key == output_key:
            raise ValueError(
                f"Cannot determine input/output fields. Record keys: {keys}\n"
                f"Expected one of input: {input_candidates[:8]} | output: {output_candidates[:8]}"
            )

    return input_key, output_key


def _parse_messages_format(records):
    """OpenAI/HF chat format: {messages: [{role, content}, ...]}"""
    user_inputs, model_outputs = [], []
    for rec in records:
        msgs = rec.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        sys = rec.get("system") or rec.get("system_prompt") or ""
        if isinstance(sys, list):
            sys = sys[0] if sys and isinstance(sys[0], str) else ""
        if not isinstance(sys, str):
            sys = ""
        for i in range(len(msgs) - 1):
            cur, nxt = msgs[i], msgs[i + 1]
            if not (isinstance(cur, dict) and isinstance(nxt, dict)):
                continue
            if cur.get("role") == "user" and nxt.get("role") == "assistant":
                cu, ca = cur.get("content", ""), nxt.get("content", "")
                if isinstance(cu, list):
                    cu = "".join(p.get("text", "") for p in cu if isinstance(p, dict))
                if isinstance(ca, list):
                    ca = "".join(p.get("text", "") for p in ca if isinstance(p, dict))
                text_in = (sys + "\n" + str(cu)).strip() if sys else str(cu).strip()
                if str(ca).strip():
                    user_inputs.append(text_in)
                    model_outputs.append(str(ca).strip())
    return user_inputs, model_outputs, "messages", "messages"


def _parse_conversations_format(records):
    """ShareGPT format: {conversations: [{from, value}, ...]}"""
    user_inputs, model_outputs = [], []
    for rec in records:
        convs = rec.get("conversations", [])
        if not isinstance(convs, list) or len(convs) < 2:
            continue
        sys = rec.get("system", "")
        if not isinstance(sys, str):
            sys = ""
        for i in range(len(convs) - 1):
            cur, nxt = convs[i], convs[i + 1]
            if not (isinstance(cur, dict) and isinstance(nxt, dict)):
                continue
            if (cur.get("from") in ("human", "user")
                    and nxt.get("from") in ("gpt", "assistant", "model", "chatbot")):
                vu = cur.get("value", cur.get("content", ""))
                va = nxt.get("value", nxt.get("content", ""))
                text_in = (sys + "\n" + str(vu)).strip() if sys else str(vu).strip()
                if str(va).strip():
                    user_inputs.append(text_in)
                    model_outputs.append(str(va).strip())
    return user_inputs, model_outputs, "conversations", "conversations"


def _parse_conversation_format(records):
    """{system, conversation: [{human, assistant}, ...]} format.

    Multi-turn conversations are expanded into multiple (user, assistant) pairs
    where user text accumulates prior context.
    """
    user_inputs, model_outputs = [], []
    for rec in records:
        system_prompt = (rec.get("system") or "").strip()
        conv = rec.get("conversation", [])
        if not isinstance(conv, list):
            continue
        context = ""
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            human = (turn.get("human") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            if not human or not assistant:
                continue
            if context:
                input_text = context + "\n" + human
            elif system_prompt:
                input_text = system_prompt + "\n" + human
            else:
                input_text = human
            user_inputs.append(input_text)
            model_outputs.append(assistant)
            if context:
                context += "\n" + human + "\n" + assistant
            else:
                context = (system_prompt + "\n" if system_prompt else "") + human + "\n" + assistant
    return user_inputs, model_outputs, "conversation", "conversation"


def _detect_uniform_system(dataset_path: str) -> str | None:
    """If every record has the same non-empty system field, return it; else None."""
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            ext = os.path.splitext(dataset_path)[1].lower()
            records = []
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            return None
            else:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        records = data["data"]
                    else:
                        for v in data.values():
                            if isinstance(v, list):
                                records = v
                                break
                        if not records:
                            records = [data]
                else:
                    return None
        sys_val = None
        for rec in records:
            if not isinstance(rec, dict):
                continue
            s = rec.get("system", "")
            if sys_val is None:
                sys_val = s
            elif sys_val != s:
                return None
        return sys_val if sys_val else None
    except Exception:
        return None


def load_json_dataset(path: str) -> Tuple[List[str], List[str], str, str]:
    """Load a JSON or JSONL dataset. Returns (user_inputs, model_outputs, input_key, output_key).

    Supported formats (auto-detected from the first record):
      1. Alpaca-like: {instruction, output[, input]}
      2. {system, conversation: [{human, assistant}, ...]}
      3. {messages: [{role, content}, ...]}  (OpenAI / HF chat)
      4. {conversations: [{from, value}, ...]}  (ShareGPT)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                records = data["data"]
            elif any(isinstance(v, list) for v in data.values()):
                for v in data.values():
                    if isinstance(v, list):
                        records = v
                        break
                else:
                    records = [data]
            else:
                records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")

    if not records:
        raise ValueError("Dataset is empty")

    first = records[0]
    if isinstance(first, dict):
        if "conversation" in first and isinstance(first["conversation"], list):
            return _parse_conversation_format(records)
        if "messages" in first and isinstance(first["messages"], list):
            return _parse_messages_format(records)
        if "conversations" in first and isinstance(first["conversations"], list):
            return _parse_conversations_format(records)

    input_key, output_key = detect_dataset_format(records)
    user_inputs, model_outputs = [], []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"Record {i} is not a dict: {type(rec)}")
        ui = rec.get(input_key, "")
        mo = rec.get(output_key, "")
        if not isinstance(ui, str):
            ui = str(ui)
        if not isinstance(mo, str):
            mo = str(mo)
        ui = ui.strip().strip('"').strip("'").strip()
        mo = mo.strip().strip('"').strip("'").strip()
        if ui and mo:
            user_inputs.append(ui)
            model_outputs.append(mo)

    return user_inputs, model_outputs, input_key, output_key


class CollateFn:
    """Picklable collate function for DataLoader multiprocessing."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        xs, ys = zip(*batch)
        max_len = max(x.size(0) for x in xs)
        x_padded = torch.full((len(xs), max_len), self.pad_id, dtype=torch.long)
        y_padded = torch.full((len(ys), max_len), self.pad_id, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            x_padded[i, :x.size(0)] = x
            y_padded[i, :y.size(0)] = y
        return x_padded, y_padded


def make_collate_fn(pad_token_id: int):
    """Returns a picklable CollateFn instance for DataLoader."""
    return CollateFn(pad_token_id)
