# inference.py -- model loading and streaming inference

import os
import io
import json
import pickle
import time
import torch
import gc
import psutil

from .config import ModelConfig
from .model import GPT
from .tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Robust model loading -- handles external .pth files with unknown classes
# ---------------------------------------------------------------------------

class _RobustUnpickler(pickle.Unpickler):
    """An Unpickler that replaces unknown classes with a stub instead of crashing."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            # Return a stub class so deserialization can continue
            stub = type(name, (), {'__module__': module, '__name__': name})
            return stub


class _RobustPickleModule:
    """A pickle-compatible module that uses _RobustUnpickler for loading."""
    Unpickler = _RobustUnpickler
    Pickler = pickle.Pickler
    loads = pickle.loads
    dumps = pickle.dumps
    PicklingError = pickle.PicklingError
    UnpicklingError = pickle.UnpicklingError
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL



def _count_params(state: dict) -> tuple:
    """Count total parameters and bytes from a state_dict (handles nested dicts)."""
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
    """Detect available CPU RAM and GPU VRAM."""
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
        info['gpu_name'] = None
        info['vram_total_gb'] = 0
        info['vram_free_gb'] = 0

    return info


def _estimate_full_memory(state: dict) -> dict:
    """Estimate memory needed to load and run inference with this model."""
    params, weight_bytes = _count_params(state)
    param_count_b = params / 1e9
    weight_gb = weight_bytes / (1024**3)
    inference_gb = weight_gb * 2.5
    inference_fp16_gb = weight_gb * 1.6

    return {
        'param_count_b': param_count_b,
        'weight_gb': weight_gb,
        'inference_gb': inference_gb,
        'inference_fp16_gb': inference_fp16_gb,
    }


def _check_memory_feasibility(state: dict) -> tuple:
    """Check if the model can fit in available memory. Returns (ok, message)."""
    hw = _detect_hardware()
    est = _estimate_full_memory(state)

    lines = []
    lines.append(f"**Hardware:** RAM total={hw['ram_total_gb']:.1f}GB, available={hw['ram_available_gb']:.1f}GB")

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
    """Load a .pth file, tolerating unknown classes from external training scripts.
    
    Loading order (most memory-efficient first):
    0. mmap=True + weights_only=True  -- zero-copy, no extra RAM
    1. weights_only=False             -- full unpickle (uses 10-15x file size)
    2. Robust pickle module           -- handles unknown classes
    3. Raw pickle + manual tensor move -- last resort
    """
    errors = []
    file_size = os.path.getsize(path)
    file_mb = file_size / (1024 * 1024)
    file_gb = file_mb / 1024
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)

    # ---- Strategy 0: mmap (zero-copy, most memory efficient) ----
    try:
        data = torch.load(path, map_location=map_location, weights_only=True, mmap=True)
        if isinstance(data, dict) and any(isinstance(v, torch.Tensor) for v in data.values()):
            return data  # raw state_dict, loaded with zero extra RAM
        if isinstance(data, torch.Tensor):
            return data
        # Not a state_dict, but loaded fine - return it
        return data
    except Exception as e:
        errors.append(f"mmap: {e}")

    # ---- Check if we have enough RAM for full unpickling ----
    # A checkpoint with optimizer states can need 10-15x file size
    # (weights + Adam states * 2 + Python overhead + temp allocations)
    estimated_need_gb = file_gb * 10

    if estimated_need_gb > avail_gb * 0.8:  # leave 20% headroom
        raise MemoryError(
            f"Insufficient RAM for full model load (mmap failed, falling back to full unpickle).\n"
            f"File size: {file_gb:.1f} GB | Estimated RAM needed: ~{estimated_need_gb:.1f} GB\n"
            f"Available RAM: {avail_gb:.1f} GB (need at least {estimated_need_gb:.1f} GB)\n\n"
            f"Suggestions:\n"
            f"1. Close other applications to free RAM\n"
            f"2. Re-export the model with: torch.save(model.state_dict(), 'model.pth')\n"
            f"   (saves ONLY weights, not optimizer states)\n"
            f"3. Convert to safetensors format"
        )

    import warnings
    warnings.warn(
        f"Loading with full unpickle (mmap unavailable). "
        f"File: {file_gb:.1f} GB | Est. need: ~{estimated_need_gb:.1f} GB | Available: {avail_gb:.1f} GB"
    )

    # ---- Strategy 1: standard torch.load ----
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except (torch.OutOfMemoryError, MemoryError, RuntimeError) as e:
        raise MemoryError(
            f"Out of memory loading model.\n"
            f"File: {file_gb:.1f} GB | Available RAM: {mem.available/(1024**3):.1f} GB\n"
            f"Try re-exporting with torch.save(model.state_dict(), ...)"
        ) from e
    except Exception as e:
        errors.append(f"standard: {e}")

    # ---- Strategy 2: robust pickle module ----
    try:
        return torch.load(
            path, map_location=map_location,
            weights_only=False,
            pickle_module=_RobustPickleModule,
        )
    except (torch.OutOfMemoryError, MemoryError, RuntimeError):
        raise
    except Exception as e:
        errors.append(f"robust pickle: {e}")

    # ---- Strategy 3: raw pickle ----
    try:
        with open(path, "rb") as f:
            data = _RobustUnpickler(f).load()
        if isinstance(data, dict):
            data = {k: v.to(map_location) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            data = data.to(map_location)
        return data
    except (torch.OutOfMemoryError, MemoryError, RuntimeError):
        raise
    except Exception as e:
        errors.append(f"raw pickle: {e}")

    raise RuntimeError(
        f"Failed to load model file with all strategies.\n"
        f"Errors: {"; ".join(errors)}\n"
        f"Try exporting as: torch.save(model.state_dict(), 'weights.pth')"
    )

def _extract_state_dict(loaded) -> dict | None:
    """Extract a state_dict from whatever was loaded."""
    if isinstance(loaded, dict):
        # Look for common keys
        for key in ('model_state_dict', 'state_dict', 'model', 'weight'):
            if key in loaded and isinstance(loaded[key], dict):
                candidate = loaded[key]
                # Verify it looks like a state_dict (has tensor values)
                if any(isinstance(v, torch.Tensor) for v in candidate.values()):
                    return candidate
        # The dict itself might be a state_dict
        if any(isinstance(v, torch.Tensor) for v in loaded.values()):
            return loaded
        return None
    elif hasattr(loaded, 'state_dict'):
        return loaded.state_dict()
    return None


def _extract_config(loaded) -> dict | None:
    """Extract a config dict from whatever was loaded."""
    if isinstance(loaded, dict):
        for key in ('config', 'model_config', 'cfg', 'args', 'hparams'):
            if key in loaded and isinstance(loaded[key], dict):
                cfg = loaded[key]
                # Check if it has model architecture keys
                if any(k in cfg for k in ('vocab_size', 'n_embd', 'hidden_size', 'n_layer')):
                    return cfg
    return None


# ---------------------------------------------------------------------------
# File/folder discovery
# ---------------------------------------------------------------------------

_MODEL_EXTENSIONS = ('.pth', '.pt', '.safetensors', '.bin', '.ckpt')


def _find_model_file(path: str) -> str:
    """Given a file or folder path, find the model file to load."""
    path = path.strip().strip('"').strip("'")

    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        candidates = []
        for fname in sorted(os.listdir(path)):
            full = os.path.join(path, fname)
            if os.path.isfile(full):
                ext = os.path.splitext(fname)[1].lower()
                if ext in _MODEL_EXTENSIONS:
                    # Prefer .safetensors > .pth > .bin > others
                    priority = {'.safetensors': 0, '.pth': 1, '.pt': 2, '.bin': 3, '.ckpt': 4}
                    candidates.append((priority.get(ext, 5), full))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        raise FileNotFoundError(
            f'No model file found in: {path}\n'
            f'Supported extensions: {", ".join(_MODEL_EXTENSIONS)}'
        )

    raise FileNotFoundError(f'Path does not exist: {path}')


def _load_config_from_folder(folder: str) -> dict | None:
    """Try to find a config.json in the model folder."""
    cfg_path = os.path.join(folder, 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _try_load_tokenizer(checkpoint: dict, model_dir: str) -> CharTokenizer | None:
    """Extract tokenizer from checkpoint or find it in model directory."""
    # 1. Try embedded tokenizer in checkpoint
    tok_data = checkpoint.get('tokenizer', None)
    if tok_data and isinstance(tok_data, dict) and 'stoi' in tok_data:
        return CharTokenizer.from_dict(tok_data)

    # 2. Try companion tokenizer files next to model
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if 'tokenizer' in fname.lower() and fname.endswith('.json'):
                try:
                    return CharTokenizer.load(os.path.join(model_dir, fname))
                except Exception:
                    pass

    return None


# ---------------------------------------------------------------------------
# Main load function
# ---------------------------------------------------------------------------

def load_model(path: str) -> tuple:
    """Load a model. Supports HuggingFace folders, .safetensors, and .pth files."""
    # Clean up any previously loaded model from GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = _find_model_file(path)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = os.path.splitext(model_path)[1].lower()
    is_dir = os.path.isdir(path)
    hf_error = None

    # ---- Strategy A: HuggingFace format ----
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
        except Exception as e:
            hf_error = str(e)

    # ---- Strategy B: AI Studio custom GPT .pth ----
    checkpoint = None
    model_state = None
    cfg_dict = None

    if ext in (".pth", ".pt", ".bin", ".ckpt"):
        loaded = _robust_torch_load(model_path, map_location=device)
        model_state = _extract_state_dict(loaded)

        if model_state is None:
            raise RuntimeError(
                f"Could not extract model weights from {model_path}. "
                f"Loaded type: {type(loaded).__name__}."
            )
        if isinstance(loaded, dict):
            checkpoint = loaded
            cfg_dict = _extract_config(loaded)

        if isinstance(loaded, dict) and loaded is not model_state and loaded is not checkpoint:
            del loaded
            gc.collect()

        if cfg_dict is None:
            cfg_dict = _load_config_from_folder(model_dir)
        if cfg_dict is None:
            cfg_dict = _infer_config_from_state(model_state)
        if cfg_dict is None:
            raise ValueError(
                "No model config found. Provide a config.json in the model folder "
                "with keys: vocab_size, n_embd, n_layer, n_head, block_size."
            )

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
            try:
                model = model.half()
            except Exception:
                pass
        model.eval()
        del model_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        tokenizer = None
        if checkpoint is not None:
            tokenizer = _try_load_tokenizer(checkpoint, model_dir)
        if tokenizer is None:
            tokenizer = _try_load_tokenizer({}, model_dir)
        if tokenizer is None:
            raise ValueError("No tokenizer found. Place a *_tokenizer.json in the model folder.")
        return model, tokenizer

    msg = f"Failed to load model from: {path}"
    if hf_error:
        msg += f"\nHuggingFace loading also failed: {hf_error}"
    msg += "\nSupported: .pth files or HuggingFace model folders with config.json"
    raise ValueError(msg)

def _infer_config_from_state(state: dict) -> dict | None:
    """Try to infer model config from state_dict tensor shapes."""
    try:
        wte_weight = state.get('transformer.wte.weight')
        if wte_weight is None:
            # Try alternative key names
            for k in state:
                if 'embed' in k.lower() and 'token' in k.lower():
                    wte_weight = state[k]
                    break
        if wte_weight is None:
            return None
        return {
            'vocab_size': wte_weight.shape[0],
            'n_embd': wte_weight.shape[1],
        }
    except Exception:
        return None


def _remap_state_keys(state: dict, config: ModelConfig) -> dict:
    """Remap common state_dict key patterns from external models to our format."""
    remapped = {}
    for key, tensor in state.items():
        new_key = key

        # HuggingFace GPT-2 style: transformer.h.X... -> our transformer.h.X...
        # Usually compatible, but handle common variations

        # Some models use 'model.' prefix
        if new_key.startswith('model.') and not new_key.startswith('model.model.'):
            new_key = new_key[6:]  # strip 'model.'

        # Some use 'gpt.' or 'transformer.' prefix (ours already uses 'transformer.')
        # Usually fine as-is

        # Handle lm_head / wte weight tying variations
        if new_key == 'lm_head.weight' and 'transformer.wte.weight' not in state:
            pass  # keep as-is

        remapped[new_key] = tensor

    return remapped


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------

def inference_stream_hf(model, tokenizer, prompt: str, history=None, max_new_tokens: int = 256, temperature: float = 0.8, top_k: int = 50):
    """Streaming inference for HuggingFace models with conversation context."""
    prompt = prompt.strip().strip('"').strip("'")
    device = next(model.parameters()).device

    # Build messages from conversation history (handles both Gradio 6 dict format and legacy tuple format)
    messages = []
    if history:
        if isinstance(history[0], dict):
            # Gradio 6+: history is a flat list of {"role":..., "content":...} dicts
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            # Legacy: history is list of (user_msg, assistant_msg) tuples
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
        top_p=0.9,
        streamer=streamer,
    )

    start_time = time.perf_counter()
    tokens_generated = [0]
    chunks = []

    def generate_thread():
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
                     history=None, max_new_tokens: int = 256, temperature: float = 0.8, top_k: int = 50):
    prompt = prompt.strip().strip('"').strip("'")

    # Build full prompt from conversation history (handles both formats)
    if history:
        parts = []
        if isinstance(history[0], dict):
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"{role}: {msg['content']}")
        else:
            for entry in history:
                parts.append(f"User: {entry[0]}")
                if len(entry) > 1 and entry[1]:
                    parts.append(f"Assistant: {entry[1]}")
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        prompt = "\n".join(parts)

    device = model.device
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    start_time = time.perf_counter()
    tokens_generated = 0
    accumulated_text = ''

    # Cache tokenizer attributes for hot-loop performance
    eos_id = tokenizer.eos_token_id
    n_special = len(tokenizer.special_tokens)
    itos = tokenizer.itos
    unk = tokenizer.unk_token

    with torch.inference_mode():
        gen = model.generate(idx, max_new_tokens=max_new_tokens,
                                            temperature=temperature, top_k=top_k)
    for token_tensor in gen:
        tokens_generated += 1
        token_id = int(token_tensor.item())
        if token_id == eos_id:
            break
        if token_id < n_special:
            continue
        ch = itos.get(token_id, unk)
        if not ch:
            continue
        accumulated_text += ch.replace('~', '\\~')
        yield accumulated_text, None

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
    yield accumulated_text, tokens_per_sec