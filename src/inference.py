# inference.py — Model loading & streaming inference for custom GPT and HF models.
# Token format (must match trainer.py):
#   input:  <bos>{user_with_system}<eos><bos>
#   output: generated tokens until <eos> (or max_new_tokens)

import os
import json
import pickle
import time
import gc
import torch
import psutil

from .config import ModelConfig
from .model import GPT
from .tokenizer import CharTokenizer

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

_MODEL_EXTENSIONS = ('.pth', '.pt', '.safetensors', '.bin', '.ckpt')


# ---------------------------------------------------------------------------
# Robust Unpickler for external .pth files with unknown classes
# ---------------------------------------------------------------------------
class _RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            return type(name, (), {'__module__': module, '__name__': name})


class _RobustPickleModule:
    Unpickler = _RobustUnpickler
    Pickler = pickle.Pickler
    loads = pickle.loads
    dumps = pickle.dumps
    PicklingError = pickle.PicklingError
    UnpicklingError = pickle.UnpicklingError
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _count_params(state: dict):
    params = 0
    param_bytes = 0
    for v in state.values():
        if isinstance(v, dict):
            p, b = _count_params(v)
            params += p
            param_bytes += b
        elif hasattr(v, 'numel') and hasattr(v, 'element_size'):
            params += v.numel()
            param_bytes += v.numel() * v.element_size()
    return params, param_bytes


def _detect_hardware() -> dict:
    info = {}
    mem = psutil.virtual_memory()
    info['ram_total_gb'] = mem.total / (1024**3)
    info['ram_available_gb'] = mem.available / (1024**3)
    info['ram_used_percent'] = mem.percent
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['vram_total_gb'] = props.total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0)
        info['vram_free_gb'] = (props.total_memory - reserved) / (1024**3)
    else:
        info.update(gpu_name=None, vram_total_gb=0, vram_free_gb=0)
    return info


def _estimate_full_memory(state: dict) -> dict:
    params, weight_bytes = _count_params(state)
    weight_gb = weight_bytes / (1024**3)
    return {
        'param_count_b': params / 1e9,
        'weight_gb': weight_gb,
        'inference_gb': weight_gb * 2.5,
        'inference_fp16_gb': weight_gb * 1.6,
    }


def _check_memory_feasibility(state: dict) -> tuple:
    hw = _detect_hardware()
    est = _estimate_full_memory(state)
    lines = [f"**Hardware:** RAM total={hw['ram_total_gb']:.1f}GB, available={hw['ram_available_gb']:.1f}GB"]
    if hw['gpu_name']:
        lines.append(f"**GPU:** {hw['gpu_name']}, VRAM total={hw['vram_total_gb']:.1f}GB, free={hw['vram_free_gb']:.1f}GB")
    lines.append(f"**Model:** {est['param_count_b']:.2f}B params, weights={est['weight_gb']:.2f}GB")
    lines.append(f"**Estimated need (fp32):** ~{est['inference_gb']:.1f}GB")
    lines.append(f"**Estimated need (fp16):** ~{est['inference_fp16_gb']:.1f}GB")
    hw_msg = '  \n'.join(lines)
    if hw['vram_free_gb'] > 0:
        if est['inference_fp16_gb'] <= hw['vram_free_gb']:
            return True, hw_msg + '  \n> Status: fits in GPU VRAM (fp16)'
        elif est['inference_gb'] <= hw['vram_free_gb']:
            return True, hw_msg + '  \n> Status: fits in GPU VRAM (fp32)'
        elif est['inference_gb'] <= hw['ram_available_gb']:
            return True, hw_msg + '  \n> Status: falling back to CPU (RAM sufficient)'
    if est['inference_gb'] <= hw['ram_available_gb']:
        return True, hw_msg + '  \n> Status: fits in system RAM'
    hw_msg += (
        f'  \n> **INSUFFICIENT MEMORY**  \n'
        f'> Available RAM: {hw["ram_available_gb"]:.1f}GB, needed: ~{est["inference_gb"]:.1f}GB  \n'
        f'> Try: close other apps, use a smaller model, or add more RAM'
    )
    return False, hw_msg


def _robust_torch_load(path: str, map_location):
    errors = []
    file_size = os.path.getsize(path)
    file_gb = file_size / (1024**3)
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)

    try:
        data = torch.load(path, map_location=map_location, weights_only=True, mmap=True)
        if isinstance(data, dict) and any(isinstance(v, torch.Tensor) for v in data.values()):
            return data
        if isinstance(data, torch.Tensor):
            return data
        return data
    except Exception as e:
        errors.append(f"mmap: {e}")

    estimated_need_gb = file_gb * 10
    if estimated_need_gb > avail_gb * 0.8:
        raise MemoryError(
            f"Insufficient RAM for full model load (mmap failed).\n"
            f"File size: {file_gb:.1f}GB | Estimated RAM needed: ~{estimated_need_gb:.1f}GB\n"
            f"Available RAM: {avail_gb:.1f}GB\n\n"
            f"Suggestions:\n"
            f"1. Close other applications\n"
            f"2. Export weights only: torch.save(model.state_dict(), 'model.pth')\n"
            f"3. Convert to safetensors"
        )

    import warnings
    warnings.warn(f"Loading with full unpickle (mmap unavailable). "
                  f"File: {file_gb:.1f}GB | Est: ~{estimated_need_gb:.1f}GB | Available: {avail_gb:.1f}GB")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except (torch.OutOfMemoryError, MemoryError) as e:
        raise MemoryError(f"OOM loading model. File: {file_gb:.1f}GB | Available: {mem.available/(1024**3):.1f}GB")
    except Exception as e:
        errors.append(f"standard: {e}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False, pickle_module=_RobustPickleModule)
    except (torch.OutOfMemoryError, MemoryError):
        raise
    except Exception as e:
        errors.append(f"robust pickle: {e}")

    try:
        with open(path, "rb") as f:
            data = _RobustUnpickler(f).load()
        if isinstance(data, dict):
            data = {k: v.to(map_location) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            data = data.to(map_location)
        return data
    except (torch.OutOfMemoryError, MemoryError):
        raise
    except Exception as e:
        errors.append(f"raw pickle: {e}")

    raise RuntimeError(f"Failed to load model with all strategies. Errors: {'; '.join(errors)}")


def _extract_state_dict(loaded):
    if isinstance(loaded, dict):
        for key in ('model_state_dict', 'state_dict', 'model', 'weight'):
            if key in loaded and isinstance(loaded[key], dict):
                candidate = loaded[key]
                if any(isinstance(v, torch.Tensor) for v in candidate.values()):
                    return candidate
        if any(isinstance(v, torch.Tensor) for v in loaded.values()):
            return loaded
        return None
    elif hasattr(loaded, 'state_dict'):
        return loaded.state_dict()
    return None


def _extract_config(loaded):
    if isinstance(loaded, dict):
        for key in ('config', 'model_config', 'cfg', 'args', 'hparams'):
            if key in loaded and isinstance(loaded[key], dict):
                cfg = loaded[key]
                if any(k in cfg for k in ('vocab_size', 'n_embd', 'hidden_size', 'n_layer')):
                    return cfg
    return None


def _find_model_file(path: str) -> str:
    path = path.strip().strip('"').strip("'").strip()
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = []
        for fname in sorted(os.listdir(path)):
            full = os.path.join(path, fname)
            if os.path.isfile(full):
                ext = os.path.splitext(fname)[1].lower()
                if ext in _MODEL_EXTENSIONS:
                    priority = {'.safetensors': 0, '.pth': 1, '.pt': 2, '.bin': 3, '.ckpt': 4}
                    candidates.append((priority.get(ext, 5), full))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        raise FileNotFoundError(f'No model file found in: {path}\nSupported: {", ".join(_MODEL_EXTENSIONS)}')
    raise FileNotFoundError(f'Path does not exist: {path}')


def _load_config_from_folder(folder: str) -> dict | None:
    cfg_path = os.path.join(folder, 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _try_load_tokenizer(checkpoint: dict, model_dir: str) -> CharTokenizer | None:
    tok_data = checkpoint.get('tokenizer', None) if isinstance(checkpoint, dict) else None
    if tok_data and isinstance(tok_data, dict) and 'stoi' in tok_data:
        return CharTokenizer.from_dict(tok_data)
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if 'tokenizer' in fname.lower() and fname.endswith('.json'):
                try:
                    return CharTokenizer.load(os.path.join(model_dir, fname))
                except Exception:
                    pass
    return None


def _infer_config_from_state(state: dict) -> dict | None:
    try:
        wte_weight = state.get('transformer.wte.weight')
        if wte_weight is None:
            for k in state:
                if 'embed' in k.lower() and 'token' in k.lower():
                    wte_weight = state[k]
                    break
        if wte_weight is None:
            return None
        return {'vocab_size': wte_weight.shape[0], 'n_embd': wte_weight.shape[1]}
    except Exception:
        return None


def _remap_state_keys(state: dict, config: ModelConfig) -> dict:
    remapped = {}
    for key, tensor in state.items():
        new_key = key
        if new_key.startswith('model.') and not new_key.startswith('model.model.'):
            new_key = new_key[6:]
        if new_key.endswith('.mlp.c_fc.weight'):
            gate_key = new_key.replace('.mlp.c_fc.weight', '.mlp.gate_proj.weight')
            up_key = new_key.replace('.mlp.c_fc.weight', '.mlp.up_proj.weight')
            remapped[gate_key] = tensor.clone()
            remapped[up_key] = tensor.clone()
            continue
        if new_key.endswith('.mlp.c_proj.weight'):
            new_key = new_key.replace('.mlp.c_proj.weight', '.mlp.down_proj.weight')
        remapped[new_key] = tensor
    return remapped


def load_model(path: str) -> tuple:
    """Load a model. Returns (model, tokenizer). system_prompt (if any) is attached
    as model._system_prompt for inference_stream to use."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = _find_model_file(path)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = os.path.splitext(model_path)[1].lower()
    is_dir = os.path.isdir(path)

    system_prompt = None

    # Strategy A: HuggingFace format
    if is_dir or ext == ".safetensors":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            hf_path = path if is_dir else model_dir
            if os.path.isdir(hf_path) and os.path.isfile(os.path.join(hf_path, "config.json")):
                model = AutoModelForCausalLM.from_pretrained(
                    hf_path, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return model, tokenizer

            if ext == ".safetensors" and not is_dir:
                from safetensors.torch import load_file as st_load
                sd = st_load(model_path, device="cpu")
                cfg_dict = _load_config_from_folder(model_dir)
                if cfg_dict is None:
                    cfg_dict = _infer_config_from_state(sd)
                if cfg_dict is not None:
                    config = ModelConfig(
                        vocab_size=cfg_dict.get("vocab_size", 512),
                        block_size=cfg_dict.get("block_size", cfg_dict.get("max_position_embeddings", 256)),
                        n_layer=cfg_dict.get("n_layer", cfg_dict.get("num_hidden_layers", 6)),
                        n_head=cfg_dict.get("n_head", cfg_dict.get("num_attention_heads", 8)),
                        n_embd=cfg_dict.get("n_embd", cfg_dict.get("hidden_size", 256)),
                        dropout=cfg_dict.get("dropout", cfg_dict.get("hidden_dropout_prob", 0.1)),
                    )
                    sd = _remap_state_keys(sd, config)
                    model = GPT(config).to(device)
                    model.load_state_dict(sd, strict=False)
                    if device.type == "cuda":
                        try: model = model.half()
                        except: pass
                    model.eval()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    tokenizer = _try_load_tokenizer({}, model_dir)
                    if tokenizer is None:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        out_dir = os.path.join(base_dir, "output")
                        avail = []
                        if os.path.isdir(out_dir):
                            for d in sorted(os.listdir(out_dir)):
                                sub = os.path.join(out_dir, d)
                                if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
                                    avail.append(d)
                        hint = "\\nAvailable models in output/: " + ", ".join(avail) if avail else ""
                        raise ValueError(f"No tokenizer found. Uploading a single .safetensors file cannot reliably locate the original folder.\\nUse the folder path input instead (e.g. D:\\\\AI Studio\\\\output\\\\MODEL_NAME).{hint}")
                    return model, tokenizer
        except Exception:
            pass

    # Strategy B: AI Studio custom GPT .pth
    if ext in (".pth", ".pt", ".bin", ".ckpt"):
        loaded = _robust_torch_load(model_path, map_location=device)
        model_state = _extract_state_dict(loaded)
        if model_state is None:
            raise RuntimeError(f"Could not extract model weights from {model_path}")

        cfg_dict = _extract_config(loaded) if isinstance(loaded, dict) else None
        if cfg_dict is None:
            cfg_dict = _load_config_from_folder(model_dir)
        if cfg_dict is None:
            cfg_dict = _infer_config_from_state(model_state)
        if cfg_dict is None:
            raise ValueError("No model config found. Provide config.json with: vocab_size, n_embd, n_layer, n_head, block_size")

        config = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 512),
            block_size=cfg_dict.get("block_size", cfg_dict.get("max_position_embeddings", 256)),
            n_layer=cfg_dict.get("n_layer", cfg_dict.get("num_hidden_layers", 6)),
            n_head=cfg_dict.get("n_head", cfg_dict.get("num_attention_heads", 8)),
            n_embd=cfg_dict.get("n_embd", cfg_dict.get("hidden_size", 256)),
            dropout=cfg_dict.get("dropout", cfg_dict.get("hidden_dropout_prob", 0.1)),
        )
        model_state = _remap_state_keys(model_state, config)

        ok, mem_msg = _check_memory_feasibility(model_state)
        if not ok:
            raise MemoryError(mem_msg)

        model = GPT(config).to(device)
        model.load_state_dict(model_state, strict=False)
        if device.type == "cuda":
            try: model = model.half()
            except: pass
        model.eval()
        # Attach system_prompt (saved with the checkpoint) for inference to use
        if isinstance(loaded, dict) and loaded.get("system_prompt"):
            system_prompt = loaded["system_prompt"]
            model._system_prompt = system_prompt
        del model_state; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        tokenizer = None
        if isinstance(loaded, dict):
            tokenizer = _try_load_tokenizer(loaded, model_dir)
        if tokenizer is None:
            tokenizer = _try_load_tokenizer({}, model_dir)
        if tokenizer is None:
            raise ValueError("No tokenizer found. Place a *_tokenizer.json in the model folder.")
        return model, tokenizer

    raise ValueError(f"Failed to load model from: {path}\nSupported: .pth files or HF model folders")


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer: CharTokenizer, user_text: str, system_prompt: str | None) -> list:
    """Build the input id list for inference: <bos>{system+user}<eos><bos>

    The model will then generate until <eos>.
    """
    user_text = user_text.strip()
    if system_prompt:
        full_user = (system_prompt.strip() + "\n" + user_text).strip()
    else:
        full_user = user_text
    ids = tokenizer.encode(full_user, add_special_tokens=True)
    # tokenizer.encode already returns <bos>...<eos>; append final <bos> as the
    # generation seed (matches trainer.py training format)
    ids.append(tokenizer.bos_token_id)
    return ids


def inference_stream_hf(model, tokenizer, prompt: str, history=None,
                        max_new_tokens: int = 256, temperature: float = 1.0, top_k: int = 50):
    """Streaming inference for HuggingFace models."""
    prompt = prompt.strip().strip('"').strip("'")
    device = next(model.parameters()).device

    messages = []
    if history:
        if isinstance(history[0], dict):
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            for entry in history:
                messages.append({"role": "user", "content": entry[0]})
                if len(entry) > 1 and entry[1]:
                    messages.append({"role": "assistant", "content": entry[1]})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)
    from transformers import TextIteratorStreamer
    from threading import Thread

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=0.9,
        use_cache=True,
        streamer=streamer,
    )

    start_time = time.perf_counter()
    tokens_generated = [0]
    chunks = []

    def generate_thread():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = Thread(target=generate_thread)
    thread.start()

    for new_text in streamer:
        tokens_generated[0] += 1
        chunks.append(new_text.replace('~', '\\~'))
        yield ''.join(chunks), None

    thread.join()
    elapsed = time.perf_counter() - start_time
    tps = tokens_generated[0] / elapsed if elapsed > 0 else 0
    yield ''.join(chunks), tps


def inference_stream(model: GPT, tokenizer: CharTokenizer, prompt: str,
                     history=None, max_new_tokens: int = 256,
                     temperature: float = 1.0, top_k: int = 50):
    """Streaming inference for custom GPT models.

    Format (must match trainer.py):
      input:  <bos>{system + user}<eos><bos>
      output: model generates tokens; stops at <eos> or max_new_tokens
    """
    prompt = prompt.strip().strip('"').strip("'").strip()
    system_prompt = getattr(model, '_system_prompt', None)

    # Truncate prompt to fit block_size (leave room for generated tokens)
    max_prompt_len = max(8, model.config.block_size - max_new_tokens - 4)
    encoded_prompt = _build_prompt(tokenizer, prompt, system_prompt)
    if len(encoded_prompt) > max_prompt_len:
        # Truncate from the beginning of the user text, keeping <bos>...<eos><bos>
        # Simplest: just drop the middle of encoded_prompt, keep both ends
        keep = max_prompt_len
        encoded_prompt = encoded_prompt[:keep]

    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=model.device)
    eos_id = tokenizer.eos_token_id
    n_special = len(tokenizer.special_tokens)
    itos = tokenizer.itos
    unk = tokenizer.unk_token

    start_time = time.perf_counter()
    tokens_generated = 0
    accumulated_text = ''

    with torch.inference_mode():
        for token_tensor in model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        ):
            tokens_generated += 1
            token_id = int(token_tensor.item())
            if token_id == eos_id:
                break
            if token_id < n_special:
                # Skip other special tokens (pad/bos/unk)
                continue
            ch = itos.get(token_id, unk)
            if not ch:
                continue
            accumulated_text += ch.replace('~', '\\~')
            yield accumulated_text, None

    elapsed = time.perf_counter() - start_time
    tps = tokens_generated / elapsed if elapsed > 0 else 0
    yield accumulated_text, tps
