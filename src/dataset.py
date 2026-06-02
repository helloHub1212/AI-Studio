# dataset.py — JSON 数据集加载与格式自动检测

import json
import torch
from torch.utils.data import Dataset
from typing import Tuple


class ConversationDataset(Dataset):
    def __init__(self, pairs: list[Tuple[str, str]], tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data: list[torch.Tensor] = []

        for user_input, model_output in pairs:
            # 将对话拼接为: <bos> user_input <eos> model_output <eos>
            user_ids = tokenizer.encode(user_input, add_special_tokens=True)
            output_ids = tokenizer.encode(model_output, add_special_tokens=True)
            all_ids = user_ids + output_ids
            if len(all_ids) > block_size:
                all_ids = all_ids[:block_size]
            self.data.append(torch.tensor(all_ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # 输入是前 n-1 个 token，目标是后 n-1 个 token
        x = seq[:-1]
        y = seq[1:]
        return x, y


def detect_dataset_format(records: list[dict]) -> Tuple[str, str]:
    if not records:
        raise ValueError('JSON 数据集为空')

    first = records[0]
    keys = list(first.keys())

    if len(keys) < 2:
        raise ValueError(
            f'每条数据至少需要两个字段，当前只有 {len(keys)} 个字段: {keys}'
        )

    # 常见键名映射
    input_candidates = ['instruction', 'input', 'prompt', 'question', 'user', 'query', 'text', 'context', 'src', 'source']
    output_candidates = ['output', 'response', 'answer', 'assistant', 'completion', 'target', 'tgt', 'reply', 'result']

    input_key = None
    output_key = None

    # 先按候选名匹配
    for ik in input_candidates:
        if ik in keys:
            input_key = ik
            break
    for ok in output_candidates:
        if ok in keys and ok != input_key:
            output_key = ok
            break

    # 如果仍有未匹配的，按顺序取第一个作为输入，最后一个作为输出
    remaining = [k for k in keys if k not in (input_key, output_key)]
    if input_key is None:
        input_key = remaining[0] if remaining else keys[0]
        if input_key in remaining:
            remaining.remove(input_key)
    if output_key is None:
        output_key = remaining[-1] if remaining else keys[-1]

    return input_key, output_key


def load_json_dataset(path: str) -> Tuple[list[str], list[str], str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        # 可能是 {\"data\": [...]} 或直接是单个对象
        if 'data' in data and isinstance(data['data'], list):
            records = data['data']
        elif any(isinstance(v, list) for v in data.values()):
            for v in data.values():
                if isinstance(v, list):
                    records = v
                    break
        else:
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f'不支持的 JSON 格式: {type(data)}')

    input_key, output_key = detect_dataset_format(records)
    user_inputs = []
    model_outputs = []

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f'第 {i} 条数据不是字典类型: {type(rec)}')
        ui = rec.get(input_key, '')
        mo = rec.get(output_key, '')
        if not isinstance(ui, str):
            ui = str(ui)
        if not isinstance(mo, str):
            mo = str(mo)
        user_inputs.append(ui.strip().strip('"').strip("'").strip())
        model_outputs.append(mo.strip().strip('"').strip("'").strip())

    return user_inputs, model_outputs, input_key, output_key
