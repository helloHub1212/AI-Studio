import gc
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "models")


def extract_pretrained_embeddings(
    model_id: str,
    char_tokenizer,
    target_dim: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    从 ModelScope/HuggingFace 模型中提取语义 embedding，
    通过 SVD 降维到 target_dim，用于初始化 wte.weight。

    返回: (vocab_size, target_dim) 的 float32 tensor
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_dir = snapshot_download(model_id, cache_dir=CACHE_DIR)

    # 加载 Qwen 的分词器和模型（只提取 embedding，用完释放）
    qwen_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    qwen_model = qwen_model.to(device)

    # 提取 embedding 表
    qwen_embed = qwen_model.get_input_embeddings().weight  # (qwen_vocab_size, qwen_dim)
    qwen_dim = qwen_embed.shape[1]
    vocab_size = len(char_tokenizer)

    # 为每个字符提取对应的 Qwen embedding
    embed_matrix = torch.zeros(vocab_size, qwen_dim, dtype=torch.float16)
    for token_id in range(len(char_tokenizer.special_tokens), vocab_size):
        char = char_tokenizer.itos[token_id]
        ids = qwen_tokenizer.encode(char, add_special_tokens=False)
        if not ids:
            continue
        embeds = qwen_embed[torch.tensor(ids, device=device)]
        embed_matrix[token_id] = embeds.mean(dim=0).cpu()

    # 释放 Qwen 模型
    del qwen_model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # SVD 降维: (vocab_size, qwen_dim) -> (vocab_size, target_dim)
    embed_matrix = embed_matrix.float()  # SVD 需要 float32
    U, S, Vt = torch.linalg.svd(embed_matrix, full_matrices=False)
    k = min(target_dim, U.shape[0], Vt.shape[0])
    reduced = U[:, :k] @ torch.diag(S[:k])
    if k < target_dim:
        padding = torch.zeros(vocab_size, target_dim - k)
        reduced = torch.cat([reduced, padding], dim=1)

    return reduced
