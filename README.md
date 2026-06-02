# AI Studio

<p align="center">
  <strong>Train &middot; Fine-tune &middot; Chat</strong><br>
  <sub>在本地训练、微调、对话 AI 模型的一站式工作台</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gradio-6.x-f97316" alt="Gradio">
  <img src="https://img.shields.io/badge/CUDA-Recommended-76b900" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## 中文文档

### 功能概览

**从零训练 AI** — 上传 JSON 对话数据集，自动识别字段格式。基于字符级分词器 + GPT 架构 Transformer（含多头注意力），流式展示训练进度，支持自定义轮数、学习率、模型规模。可从 Qwen 等大模型中提取语义 embedding 作为初始知识，显著提升训练效果。训练完成后保存为 `.pth` 模型文件。

**大模型微调** — 从 ModelScope 自动下载开源大模型，支持全量微调和 LoRA 两种模式。由 HuggingFace Trainer 驱动，支持 bf16 混合精度，结果保存为标准 HuggingFace 格式。

**使用 AI 对话** — 支持加载自训练 `.pth` 模型和 HuggingFace `.safetensors` 模型。流式逐 token 输出，显示 tokens/sec 推理速度，具备多轮对话上下文记忆。自动检测显存/内存，不足时阻止加载并提示。

**性能监控** — 实时显示 CPU / RAM / GPU / VRAM 利用率，暗色渐变状态栏。

### 界面截图

#### 训练 AI
> 从 JSON 数据集从零训练，流式显示每轮 loss 和学习率变化

![Train Tab](./docs/tab-train.png)

![Training Demo](./docs/demo-train.gif)

#### 微调大模型
> 从 ModelScope 下载模型，支持 LoRA / 全量微调

![Fine-tune Tab](./docs/tab-finetune.png)

#### 对话
> 流式逐 token 输出，多轮上下文记忆，显示推理速度

![Chat Tab](./docs/tab-chat.png)

![Chat Demo](./docs/demo-chat.gif)

#### 性能监控
> 实时 CPU / RAM / GPU / VRAM 使用率

![Monitor Bar](./docs/monitor.png)

### 快速开始

**环境要求**

| 项目 | 最低版本 | 推荐 |
|------|----------|------|
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.0+ (CUDA) | 2.5+ cu130 |
| CUDA | 11.8+ | 12.4+ |
| RAM | 8 GB | 16 GB+ |
| VRAM (训练) | 4 GB | 12 GB+ |

**安装**

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**启动**

```bash
python main.py
```

或双击 `start.bat`（Windows），浏览器自动打开 `http://127.0.0.1:7860`。

### 使用指南

**从零训练 AI**

1. 准备 JSON 数据集（支持任意键名，自动识别）：
```json
[
  {"input": "你好", "output": "你好！有什么可以帮助你的？"},
  {"input": "今天天气怎么样", "output": "抱歉，我无法获取实时天气信息。"}
]
```
2. 在 **Train AI** 标签页上传 `.json` 文件
3. 调整训练轮数 (Epochs)、学习率 (Learning Rate)、模型规模 (small / medium / large / xlarge)
4. 默认勾选「使用预训练 Embedding」，从 Qwen3.5-0.8B 提取语义知识初始化模型（可自行指定源模型 ID）
5. 点击 **Start Training**，实时查看训练日志
6. 训练完成后输入模型名称，点击 **Save** 保存到 `./output/`

**大模型微调**

1. 在 **Fine-tune** 标签页输入 ModelScope 模型 ID，例如 `Qwen/Qwen2.5-0.5B`
2. 上传 JSON 训练数据集
3. 选择微调模式：勾选 LoRA（推荐，r=8, alpha=16）或取消勾选进行全量微调
4. 设置训练参数后点击 **Start Fine-tuning**
5. 模型自动下载到 `./cache/models/`，训练完成后保存到 `./output/`，缓存自动清理

**使用 AI 对话**

1. 在 **Use AI** 标签页上传模型文件或输入模型文件夹路径
2. 点击 **Load Model**，等待加载完成
3. 在对话框输入消息，模型流式输出回复，底部显示 tokens/sec
4. 加载新模型时自动清空历史上下文

### 项目结构

```
ai-studio/
├── main.py              # Gradio UI 主入口
├── start.bat            # Windows 一键启动
├── requirements.txt     # Python 依赖
├── src/
│   ├── model.py         # GPT Transformer 模型 (多头注意力)
│   ├── trainer.py       # 从零训练循环 + 流式进度
│   ├── inference.py     # 模型加载 + 流式推理 + 显存检测
│   ├── finetune.py      # ModelScope 下载 + HF Trainer 微调
?   ??? extract_embeddings.py  # ???????? embedding
│   ├── dataset.py       # JSON 数据集加载 + 格式识别
│   ├── tokenizer.py     # 字符级分词器
│   └── config.py        # 模型超参数配置
├── output/              # 训练/微调后的模型
├── cache/models/        # ModelScope 下载缓存
└── docs/                # 截图与文档图片
```

### 模型架构

自定义 GPT 基于 Decoder-Only Transformer：

- **CausalSelfAttention** — 因果多头自注意力（下三角掩码）
- **MLP** — GELU + 4x 前馈网络
- **TransformerBlock** — Pre-LN 残差连接
- **GPT** — 位置编码 + Token 嵌入 + N 层 Transformer + LM Head
- 权重绑定：wte ↔ lm_head

| 规模 | n_layer | n_head | n_embd | block_size |
|------|---------|--------|--------|------------|
| Small | 4 | 4 | 128 | 128 |
| Medium | 6 | 8 | 256 | 256 |
| Large | 8 | 8 | 512 | 512 |
| XLarge | 12 | 12 | 768 | 768 |

### 注意事项

- 首次微调从 ModelScope 下载模型，需要网络连接
- 加载大模型时自动检测显存/内存，不足时阻止加载并提示
- `weights_only=False` 用于加载自定义 `.pth` 文件 —— 仅在信任来源的模型文件上使用
- 训练自定义 GPT 时 block_size 自动适配数据集最长对话（上限 2048）

### 许可

MIT License

---

## English Documentation

### Features

**Train from Scratch** — Upload a JSON conversation dataset and the system auto-detects field names. Built on a character-level tokenizer with a GPT-style Transformer (multi-head attention). Streaming training progress with customizable epochs, learning rate, and model size. Saves as `.pth`.

**Fine-tune** — Auto-download open-source models from ModelScope. Supports both full fine-tuning and LoRA. Powered by HuggingFace Trainer with bf16 mixed precision. Output saved in standard HuggingFace format.

**Chat** — Load custom `.pth` or HuggingFace `.safetensors` models. Streaming token-by-token output with tokens/sec display and multi-turn conversation context. Auto-detects GPU/CPU memory and prevents loading if insufficient.

**Performance Monitor** — Real-time CPU / RAM / GPU / VRAM utilization with a dark gradient status bar.

### Screenshots

#### Train from Scratch
> Streaming training progress with per-epoch loss and learning rate

![Train Tab](./docs/tab-train.png)

![Training Demo](./docs/demo-train.gif)

#### Fine-tune
> Download models from ModelScope with LoRA / full fine-tuning support

![Fine-tune Tab](./docs/tab-finetune.png)

#### Chat
> Streaming token-by-token output with context memory and tokens/sec display

![Chat Tab](./docs/tab-chat.png)

![Chat Demo](./docs/demo-chat.gif)

#### Monitor
> Real-time CPU / RAM / GPU / VRAM usage

![Monitor Bar](./docs/monitor.png)

### Quick Start

**Requirements**

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.0+ (CUDA) | 2.5+ cu130 |
| CUDA | 11.8+ | 12.4+ |
| RAM | 8 GB | 16 GB+ |
| VRAM | 4 GB | 12 GB+ |

**Install**

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Launch**

```bash
python main.py
```

Or double-click `start.bat` (Windows). Opens `http://127.0.0.1:7860` automatically.

### User Guide

**Train from Scratch**

1. Prepare a JSON dataset (any key names, auto-detected):
```json
[
  {"input": "Hello", "output": "Hi there! How can I help?"},
  {"input": "What is AI?", "output": "AI stands for Artificial Intelligence..."}
]
```
2. Upload the `.json` file in the **Train AI** tab
3. Adjust epochs, learning rate, and model size (small / medium / large / xlarge)
4. Enable "Pretrained Embedding" to initialize from a source model (default: Qwen3.5-0.8B)
5. Click **Start Training** and watch real-time progress
6. Name your model and click **Save** -> stored in `./output/`

**Fine-tune**

1. Enter a ModelScope model ID in the **Fine-tune** tab, e.g. `Qwen/Qwen2.5-0.5B`
2. Upload a JSON training dataset
3. Choose mode: check LoRA (recommended, r=8, alpha=16) or uncheck for full fine-tuning
4. Set parameters and click **Start Fine-tuning**
5. Model downloads to `./cache/models/`, result saved to `./output/`, cache auto-cleaned

**Chat**

1. Upload a model file or enter a folder path in the **Use AI** tab
2. Click **Load Model** and wait for it to load
3. Type a message — the model streams its reply with tokens/sec shown below
4. Loading a new model automatically clears conversation history

### Project Structure

```
ai-studio/
├── main.py              # Gradio UI entry point
├── start.bat            # Windows launcher
├── requirements.txt     # Python dependencies
├── src/
│   ├── model.py         # GPT Transformer (multi-head attention)
│   ├── trainer.py       # Training loop + streaming progress
│   ├── inference.py     # Model loading + streaming inference + memory check
│   ├── finetune.py      # ModelScope download + HF Trainer fine-tuning
?   ??? extract_embeddings.py  # Pretrained embedding extraction
│   ├── dataset.py       # JSON dataset loader + format detection
│   ├── tokenizer.py     # Character-level tokenizer
│   └── config.py        # Model hyperparameters
├── output/              # Trained / fine-tuned models
├── cache/models/        # ModelScope download cache
└── docs/                # Screenshots & images
```

### Architecture

Custom GPT built on Decoder-Only Transformer:

- **CausalSelfAttention** — Causal multi-head self-attention (lower-triangular mask)
- **MLP** — GELU + 4x feed-forward
- **TransformerBlock** — Pre-LN residual connection
- **GPT** — Positional encoding + Token embedding + N Transformer layers + LM Head
- Weight tying: wte <-> lm_head

| Size | n_layer | n_head | n_embd | block_size |
|------|---------|--------|--------|------------|
| Small | 4 | 4 | 128 | 128 |
| Medium | 6 | 8 | 256 | 256 |
| Large | 8 | 8 | 512 | 512 |
| XLarge | 12 | 12 | 768 | 768 |

### Notes

- First fine-tune run downloads the model from ModelScope (requires internet)
- Memory auto-detection prevents loading models that exceed available RAM/VRAM
- `weights_only=False` is used for custom `.pth` files — only use with trusted sources
- Training auto-adjusts block_size to the longest conversation in the dataset (max 2048)

### License

MIT License

---

<p align="center">
  <sub>Made with PyTorch & Gradio</sub>
</p>