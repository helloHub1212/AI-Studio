# tokenizer.py — Character-level Tokenizer with save/load

import json
import os
from collections import OrderedDict
from typing import List, Dict, Any

SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '་']


class CharTokenizer:
    def __init__(self, texts: List[str] | None = None):
        self.special_tokens = SPECIAL_TOKENS
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.unk_token = '་'

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.stoi: Dict[str, int] = OrderedDict()
        self.itos: Dict[int, str] = OrderedDict()

        for i, tok in enumerate(self.special_tokens):
            self.stoi[tok] = i
            self.itos[i] = tok

        if texts:
            self._build_vocab(texts)

    def _build_vocab(self, texts: List[str]):
        chars = set()
        for text in texts:
            for ch in text:
                chars.add(ch)
        for ch in sorted(chars, key=lambda c: ord(c)):
            if ch not in self.stoi:
                idx = len(self.stoi)
                self.stoi[ch] = idx
                self.itos[idx] = ch

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for tid in ids:
            if skip_special_tokens and tid < len(self.special_tokens):
                continue
            chars.append(self.itos.get(tid, self.unk_token))
        return ''.join(chars)

    def save(self, path: str):
        state = {
            'stoi': dict(self.stoi),
            'itos': {str(k): v for k, v in self.itos.items()},
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        tokenizer = cls()
        tokenizer.stoi = OrderedDict(state['stoi'])
        tokenizer.itos = OrderedDict({int(k): v for k, v in state['itos'].items()})
        return tokenizer

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stoi': dict(self.stoi),
            'itos': {str(k): v for k, v in self.itos.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharTokenizer':
        tokenizer = cls()
        tokenizer.stoi = OrderedDict(data['stoi'])
        tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
        return tokenizer

    def __len__(self) -> int:
        return self.vocab_size