# AI Studio - Train, Fine-tune, Chat
import os, sys, json, time, gc, threading, gradio as gr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import load_json_dataset, ConversationDataset
from src.tokenizer import CharTokenizer
from src.trainer import save_model as save_trained_model, train_model_stream
from src.inference import load_model, inference_stream, inference_stream_hf
from src.finetune import fine_tune_stream

import torch, psutil
try:
    import pynvml; pynvml.nvmlInit(); _pynvml_ok = True
except Exception:
    _pynvml_ok = False

CUSTOM_CSS = "*{font-family:'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}\n.gradio-container{width:860px;min-width:860px;max-width:860px;margin:0 auto}\n#app-title{text-align:center;font-size:2.2rem;font-weight:300;letter-spacing:.05em;color:#2c2c2c;margin-bottom:.2rem}\n#app-subtitle{text-align:center;font-size:.9rem;color:#999;margin-bottom:1.5rem;font-weight:300}\n.tab-nav{justify-content:center;border-bottom:none}\n.tab-nav button{font-size:.9rem;padding:.5rem 1.5rem;border-radius:6px;color:#666;border:none;background:transparent;transition:all .2s ease}\n.tab-nav button.selected{color:#333;background:#f0f0f0;font-weight:500}\n.progress-box textarea{font-family:'Cascadia Code',Consolas,monospace;font-size:.82rem;line-height:1.6;background:#fafafa}\n.save-panel{background:#f9fafb;border-radius:8px;padding:1.5rem;border:1px solid #e5e7eb;margin-top:.5rem}\nfooter{text-align:center;color:#ccc;font-size:.7rem;margin-top:1.5rem}\n.status-ok{color:#059669}\n.status-err{color:#dc2626}\n.monitor-bar{display:flex;gap:12px;padding:10px 16px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);border-radius:10px;margin-top:16px;align-items:center}\n.monitor-item{flex:1;text-align:center}\n.monitor-label{font-size:.65rem;color:#8892b0;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}\n.monitor-value{font-size:.95rem;font-weight:500;color:#e6f1ff}\n.monitor-bar-fill{height:3px;border-radius:2px;margin-top:4px;transition:width .5s ease}\n.monitor-cpu .monitor-bar-fill{background:linear-gradient(90deg,#64ffda,#00bfa5)}\n.monitor-ram .monitor-bar-fill{background:linear-gradient(90deg,#82b1ff,#448aff)}\n.monitor-gpu .monitor-bar-fill{background:linear-gradient(90deg,#b388ff,#7c4dff)}\n.monitor-vram .monitor-bar-fill{background:linear-gradient(90deg,#ff8a80,#ff5252)}"

def monitor_html():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    ram_used = psutil.virtual_memory().used / (1024**3)
    ram_total = psutil.virtual_memory().total / (1024**3)
    gpu_u = 0; vram_pct = 0; gpu_str = 'N/A'; vram_str = 'N/A'
    if _pynvml_ok:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            vram_i = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_pct = vram_i.used / vram_i.total * 100
            gpu_str = f'{gpu_u}%'
            vram_str = f'{vram_pct:.0f}% ({vram_i.used/(1024**3):.1f}/{vram_i.total/(1024**3):.1f}G)'
        except Exception:
            pass
    return f'<div class="monitor-bar"><div class="monitor-item monitor-cpu"><div class="monitor-label">CPU</div><div class="monitor-value">{cpu}%</div><div class="monitor-bar-fill" style="width:{cpu}%"></div></div><div class="monitor-item monitor-ram"><div class="monitor-label">RAM</div><div class="monitor-value">{ram}% ({ram_used:.1f}/{ram_total:.1f}G)</div><div class="monitor-bar-fill" style="width:{ram}%"></div></div><div class="monitor-item monitor-gpu"><div class="monitor-label">GPU</div><div class="monitor-value">{gpu_str}</div><div class="monitor-bar-fill" style="width:{gpu_u}%"></div></div><div class="monitor-item monitor-vram"><div class="monitor-label">VRAM</div><div class="monitor-value">{vram_str}</div><div class="monitor-bar-fill" style="width:{vram_pct:.0f}%"></div></div></div>'

def _do_train(dataset_path, epochs, learning_rate, model_size, use_pretrained=True, source_model_id="Qwen/Qwen3.5-0.8B", use_checkpoint=False, grad_accum_steps=1, num_workers=0, warmup_ratio=0.05, eval_ratio=0.0, compile_model=True, early_stopping=True, early_stopping_patience=3, early_stopping_threshold=0.005):
    if dataset_path is None:
        yield 'Please upload a JSON dataset file first', gr.update(visible=False), None, None, '', gr.update(interactive=True)
        return
    try:
        yield 'Loading dataset...', gr.update(visible=False), None, None, '', gr.update(interactive=False)
        user_inputs, model_outputs, input_key, output_key = load_json_dataset(dataset_path)
        # Detect a uniform system_prompt across the dataset (for conversation format)
        from src.dataset import _detect_uniform_system
        system_prompt = _detect_uniform_system(dataset_path)
        yield f'Dataset: {len(user_inputs)} conversations (input={input_key}, output={output_key})' + (f' | system: "{system_prompt[:40]}..."' if system_prompt else ''), gr.update(visible=False), None, None, '', gr.update(interactive=False)
        if user_inputs and model_outputs:
            preview = "\n".join(
                f"**样本 {i+1}**\n"
                f"  • Input:  `{user_inputs[i][:120]}{'...' if len(user_inputs[i])>120 else ''}`\n"
                f"  • Output: `{model_outputs[i][:120]}{'...' if len(model_outputs[i])>120 else ''}`"
                for i in range(min(2, len(user_inputs)))
            )
            yield f"数据预览:\n{preview}", gr.update(visible=False), None, None, '', gr.update(interactive=False)
        yield 'Building tokenizer...', gr.update(visible=False), None, None, '', gr.update(interactive=False)
        all_texts = [f'{u} {o}' for u, o in zip(user_inputs, model_outputs)]
        tokenizer = CharTokenizer(all_texts)
        yield f'Tokenizer: vocab {tokenizer.vocab_size}', gr.update(visible=False), None, None, '', gr.update(interactive=False)
        from src.config import ModelConfig
        base_size = model_size.split()[0]
        size_map = {'small': ModelConfig.small(), 'medium': ModelConfig.medium(), 'large': ModelConfig.large(), 'xlarge': ModelConfig.xlarge(), 'max': ModelConfig.max()}
        config = size_map.get(base_size, ModelConfig.small())
        dataset = ConversationDataset(list(zip(user_inputs, model_outputs)), tokenizer, config.block_size)
        yield f'Dataset: {len(dataset)} samples ready', gr.update(visible=False), None, None, '', gr.update(interactive=False)
        pretrained_weight = None
        if use_pretrained and source_model_id and source_model_id.strip():
            yield f'Extracting pretrained embeddings from {source_model_id.strip()}...', gr.update(visible=False), None, None, '', gr.update(interactive=False)
            try:
                from src.extract_embeddings import extract_pretrained_embeddings
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                pretrained_weight = extract_pretrained_embeddings(source_model_id.strip(), tokenizer, config.n_embd, dev)
                yield f'Pretrained embeddings ready: vocab={pretrained_weight.shape[0]}, dim={pretrained_weight.shape[1]}', gr.update(visible=False), None, None, '', gr.update(interactive=False)
            except Exception as ex:
                yield f'Pretrained extraction failed, using random init: {ex}', gr.update(visible=False), None, None, '', gr.update(interactive=False)
        model = None
        for progress_msg in train_model_stream(dataset=dataset, tokenizer=tokenizer, epochs=int(epochs), learning_rate=float(learning_rate), batch_size=16, model_size=base_size, pretrained_embed_weight=pretrained_weight, use_checkpoint=use_checkpoint, grad_accum_steps=int(grad_accum_steps), num_workers=int(num_workers), warmup_ratio=float(warmup_ratio), eval_ratio=float(eval_ratio), compile_model=compile_model, early_stopping=bool(early_stopping), early_stopping_patience=int(early_stopping_patience), early_stopping_threshold=float(early_stopping_threshold)):
            if isinstance(progress_msg, str):
                if progress_msg.startswith('###'):
                    line = ' '.join(progress_msg.replace('### ', '').splitlines())
                    yield line, gr.update(visible=False), None, None, '', gr.update(interactive=False)
            else:
                model = progress_msg
        if model is not None and hasattr(model, 'cpu'):
            model = model.cpu()
        # Attach system_prompt to model so save can persist it
        if model is not None and system_prompt:
            model._system_prompt = system_prompt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield 'Training complete! Name your model below and save.', gr.update(visible=True), model, tokenizer, '', gr.update(interactive=True)
    except Exception as e:
        import traceback
        yield f'Error: {str(e)}\n{traceback.format_exc()}', gr.update(visible=False), None, None, '', gr.update(interactive=True)

def _do_save(model_name, model, tokenizer):
    if not model_name or not model_name.strip(): return 'Please enter a valid model name'
    if model is None: return 'No model to save, please train first'
    try:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        system_prompt = getattr(model, '_system_prompt', None)
        model_path = save_trained_model(model, tokenizer, output_dir, model_name.strip(), system_prompt=system_prompt)
        if model is not None and hasattr(model, 'cpu'):
            model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f'Saved to: {model_path}'
    except Exception as e:
        return f'Save failed: {str(e)}'

def _do_load_model(model_file):
    if model_file is None: return None, None, '<span class="status-err">Please upload a model file</span>'
    try:
        model, tokenizer = load_model(model_file)
        device = 'GPU' if next(model.parameters()).is_cuda else 'CPU'
        return model, tokenizer, f'<span class="status-ok">Model loaded ({device})</span>'
    except MemoryError as e: return None, None, f'<span class="status-err">{str(e)}</span>'
    except Exception as e: return None, None, f'<span class="status-err">Load failed: {str(e)}</span>'

def _chat_fn(message, history, model, tokenizer, temperature, max_context_turns):
    message = message.strip().strip('"').strip("'").strip()
    if model is None or tokenizer is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please import a model first"})
        yield history, ''
        return
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    final_tps = 0.0
    last_yield = time.perf_counter()
    try:
        from transformers import PreTrainedModel
        is_hf = isinstance(model, PreTrainedModel)
        infer_fn = inference_stream_hf if is_hf else inference_stream

        # Build the prompt. Custom GPT models: use model._system_prompt (saved with checkpoint);
        # HF models: pass chat-template history.
        if is_hf:
            full_history = history[:-2]
            if max_context_turns > 0 and len(full_history) > max_context_turns * 2:
                system_msgs = [m for m in full_history if m.get("role") == "system"]
                other_msgs = [m for m in full_history if m.get("role") != "system"]
                keep_msgs = other_msgs[-(max_context_turns * 2):]
                model_history = system_msgs + keep_msgs
            else:
                model_history = full_history
            infer_kwargs = dict(history=model_history)
        else:
            # Custom GPT: build prompt from current message; system_prompt is auto-prepended
            # by inference_stream via model._system_prompt. Multi-turn context is currently
            # not supported for custom GPT (model trained on single-turn).
            infer_kwargs = dict()

        for acc_text, tps in infer_fn(model, tokenizer, message,
                                       max_new_tokens=256,
                                       temperature=temperature,
                                       top_k=50,
                                       **infer_kwargs):
            if tps is not None:
                final_tps = tps
            history[-1]["content"] = acc_text
            now = time.perf_counter()
            if now - last_yield >= 0.03:
                last_yield = now
                yield history, ''
        yield history, ''
        if final_tps > 0:
            history[-1]["content"] = f'{history[-1]["content"]}\n\n<small style="color:#bbb;">{final_tps:.1f} tokens/sec</small>'
            yield history, ''
    except Exception as e:
        import traceback
        history[-1]["content"] = f'Generation error: {str(e)}\n\n`\n{"".join(traceback.format_exception(type(e), e, e.__traceback__))}\n`'
        yield history, ''

def _do_finetune(model_id, json_path, epochs, batch_size, lr, use_lora, lora_r, lora_alpha, save_name, use_4bit=False, use_checkpoint=True, grad_accum_steps=1, num_workers=0, warmup_ratio=0.05, eval_ratio=0.0, max_seq_length=512, packing=False, early_stopping=True, early_stopping_patience=3, early_stopping_threshold=0.005):
    if not model_id or not model_id.strip():
        yield 'Please enter a ModelScope model ID'
        return
    if not json_path:
        yield 'Please upload a JSON dataset file'
        return
    if not save_name or not save_name.strip():
        save_name = model_id.strip().replace('/', '_')
    try:
        for msg in fine_tune_stream(
            model_id=model_id.strip(), json_path=json_path, epochs=int(epochs),
            batch_size=int(batch_size), learning_rate=float(lr),
            use_lora=use_lora, lora_r=int(lora_r), lora_alpha=int(lora_alpha),
            save_name=save_name.strip(), use_4bit=use_4bit, use_checkpoint=use_checkpoint,
            grad_accum_steps=int(grad_accum_steps), num_workers=int(num_workers),
            warmup_ratio=float(warmup_ratio), eval_ratio=float(eval_ratio),
            max_seq_length=int(max_seq_length), packing=packing,
            early_stopping=bool(early_stopping),
            early_stopping_patience=int(early_stopping_patience),
            early_stopping_threshold=float(early_stopping_threshold),
        ):
            if msg.startswith('SAVED:'):
                path = msg[6:]
                yield f'Fine-tuning complete!\nModel saved to: {path}'
            else:
                yield msg
    except Exception as e:
        import traceback
        yield f'Error: {str(e)}\n{traceback.format_exc()}'

def build_ui():
    with gr.Blocks(title='AI Studio') as demo:
        model_state = gr.State(None)
        tokenizer_state = gr.State(None)
        chat_model_state = gr.State(None)
        chat_tokenizer_state = gr.State(None)
        gr.HTML('<div id="app-title">AI Studio</div><div id="app-subtitle">Train &middot; Fine-tune &middot; Chat</div>')
        with gr.Tabs(elem_classes='tab-nav'):
            # ---- Train AI ----
            with gr.TabItem('Train AI'):
                with gr.Row():
                    dataset_file = gr.File(label='Dataset File (JSON/JSONL)', file_types=['.json','.jsonl'], type='filepath')
                with gr.Row(equal_height=True):
                    epochs_slider = gr.Slider(1, 100, value=10, step=1, label='Epochs')
                    lr_slider = gr.Slider(1e-5, 1e-2, value=3e-4, label='Learning Rate')
                    size_dropdown = gr.Dropdown(choices=['small (4L-128d)', 'medium (6L-256d)', 'large (8L-512d)', 'xlarge (12L-768d)', 'max (24L-1024d)'], value='small (4L-128d)', label='Model Size')
                with gr.Row(equal_height=True):
                    use_pretrained = gr.Checkbox(label='使用预训练Embedding', value=True)
                    source_model = gr.Textbox(label='源模型ID', value='Qwen/Qwen3.5-0.8B', placeholder='e.g. Qwen/Qwen3.5-0.8B', scale=2)
                with gr.Row(equal_height=True):
                    use_checkpoint = gr.Checkbox(label='梯度检查点 (省显存换速度)', value=False)
                    grad_accum = gr.Slider(1, 32, value=1, step=1, label='梯度累积步数', scale=1)
                    num_workers_inp = gr.Slider(0, 8, value=0, step=1, label='DataLoader Workers (Win建议0)', scale=1)
                with gr.Row(equal_height=True):
                    warmup_ratio_inp = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label='Warmup 比例', scale=1)
                    eval_ratio_inp = gr.Slider(0.0, 0.1, value=0.1, step=0.01, label='验证集比例（≥0.01 才开启早停）', scale=1)
                    compile_model_inp = gr.Checkbox(label='torch.compile 加速', value=True, scale=1)
                with gr.Row(equal_height=True):
                    early_stopping_inp = gr.Checkbox(label='自动早停（需开启验证集）', value=True, scale=1)
                    early_stopping_patience_inp = gr.Slider(1, 10, value=3, step=1, label='早停 Patience', scale=1)
                    early_stopping_threshold_inp = gr.Slider(0.0, 0.05, value=0.005, step=0.001, label='早停 Threshold (改善幅度)', scale=1)
                train_btn = gr.Button('Start Training', variant='primary', size='lg')
                progress_box = gr.Textbox(label='Training Log', lines=12, max_lines=12, interactive=False, elem_classes='progress-box')
                with gr.Group(visible=False, elem_classes='save-panel') as save_panel:
                    gr.Markdown('### Save Model')
                    with gr.Row():
                        name_input = gr.Textbox(label='Model Name', placeholder='e.g. my_model', scale=3)
                        save_btn = gr.Button('Save', variant='primary', scale=1)
                    save_status = gr.Markdown('')
                train_btn.click(fn=_do_train, inputs=[dataset_file, epochs_slider, lr_slider, size_dropdown, use_pretrained, source_model, use_checkpoint, grad_accum, num_workers_inp, warmup_ratio_inp, eval_ratio_inp, compile_model_inp, early_stopping_inp, early_stopping_patience_inp, early_stopping_threshold_inp], outputs=[progress_box, save_panel, model_state, tokenizer_state, name_input, train_btn])
                save_btn.click(fn=_do_save, inputs=[name_input, model_state, tokenizer_state], outputs=[save_status])
            # ---- Fine-tune ----
            with gr.TabItem('Fine-tune'):
                with gr.Row():
                    model_id_input = gr.Textbox(label='ModelScope Model ID', placeholder='e.g. Qwen/Qwen2.5-0.5B', scale=2)
                    ft_json = gr.File(label='Dataset (JSON/JSONL)', file_types=['.json','.jsonl'], type='filepath', scale=1)
                with gr.Row(equal_height=True):
                    ft_epochs = gr.Slider(1, 20, value=3, step=1, label='Epochs')
                    ft_batch = gr.Slider(1, 16, value=2, step=1, label='Batch Size')
                    ft_lr = gr.Slider(1e-6, 1e-3, value=2e-5, label='Learning Rate')
                with gr.Row():
                    use_lora_check = gr.Checkbox(label='Use LoRA', value=True)
                    ft_lora_r = gr.Slider(4, 64, value=8, step=4, label='LoRA r', visible=True)
                    ft_lora_alpha = gr.Slider(8, 128, value=16, step=4, label='LoRA alpha', visible=True)
                with gr.Row():
                    use_4bit_check = gr.Checkbox(label='4-bit QLoRA (省更多显存)', value=False)
                    ft_use_checkpoint = gr.Checkbox(label='梯度检查点 (省显存)', value=True)
                    ft_save_name = gr.Textbox(label='Save Name', placeholder='my_finetuned_model', scale=2)
                    ft_btn = gr.Button('Start Fine-tuning', variant='primary', scale=1)
                with gr.Row(equal_height=True):
                    ft_grad_accum = gr.Slider(1, 32, value=1, step=1, label='梯度累积步数', scale=1)
                    ft_num_workers = gr.Slider(0, 8, value=0, step=1, label='DataLoader Workers (Win建议0)', scale=1)
                    ft_warmup = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label='Warmup 比例', scale=1)
                    ft_eval_ratio = gr.Slider(0.0, 0.1, value=0.1, step=0.01, label='验证集比例（≥0.01 才开启早停）', scale=1)
                with gr.Row(equal_height=True):
                    ft_max_len = gr.Slider(128, 4096, value=512, step=128, label='最大序列长度', scale=2)
                    ft_packing = gr.Checkbox(label='序列打包 (Packing 提升利用率)', value=False, scale=1)
                with gr.Row(equal_height=True):
                    ft_early_stopping = gr.Checkbox(label='自动早停（需开启验证集）', value=True, scale=1)
                    ft_early_stopping_patience = gr.Slider(1, 10, value=3, step=1, label='早停 Patience', scale=1)
                    ft_early_stopping_threshold = gr.Slider(0.0, 0.05, value=0.005, step=0.001, label='早停 Threshold', scale=1)
                ft_output = gr.Textbox(label='Status', lines=10, max_lines=10, interactive=False, elem_classes='progress-box')
                def toggle_lora(v):
                    return gr.update(visible=v), gr.update(visible=v)
                use_lora_check.change(fn=toggle_lora, inputs=use_lora_check, outputs=[ft_lora_r, ft_lora_alpha])
                ft_btn.click(fn=_do_finetune, inputs=[model_id_input, ft_json, ft_epochs, ft_batch, ft_lr, use_lora_check, ft_lora_r, ft_lora_alpha, ft_save_name, use_4bit_check, ft_use_checkpoint, ft_grad_accum, ft_num_workers, ft_warmup, ft_eval_ratio, ft_max_len, ft_packing, ft_early_stopping, ft_early_stopping_patience, ft_early_stopping_threshold], outputs=[ft_output])
            # ---- Use AI ----
            with gr.TabItem('Use AI'):
                with gr.Row():
                    model_file = gr.File(label='Model (.pth/.safetensors)', file_types=['.pth','.safetensors'], type='filepath', scale=2)
                    model_folder = gr.Textbox(label='Or Folder Path', placeholder='D:\\AI Studio\\output\\MODEL_NAME', scale=1)
                load_btn = gr.Button('Load Model', variant='primary')
                load_status = gr.Markdown('Waiting for model import...')
                with gr.Row(equal_height=True):
                    temperature_slider = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label='Temperature', scale=2)
                    max_context_turns = gr.Slider(0, 20, value=5, step=1, label='Max Context Turns (0=all)', scale=2)
                chatbot = gr.Chatbot(height=440, placeholder='Import a model to start chatting...')
                chat_input = gr.Textbox(placeholder='Type a message and press Enter...', container=False)
                def _on_load(file_path, folder_path):
                    if chat_model_state.value is not None:
                        try:
                            if hasattr(chat_model_state.value, 'cpu'):
                                chat_model_state.value.cpu()
                        except Exception:
                            pass
                    chat_model_state.value = None
                    chat_tokenizer_state.value = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    path = (file_path or folder_path or "").strip().strip(chr(34)).strip(chr(39))
                    if not path or not path.strip():
                        return gr.update(), gr.update(), '<span class="status-err">Please select a file or enter a folder path</span>', gr.update()
                    model, tok, status = _do_load_model(path.strip())
                    if model is not None:
                        if tok is None:
                            return model, model, status, []
                        return model, tok, status, []
                    else:
                        return gr.update(), gr.update(), status, gr.update()
                load_btn.click(fn=_on_load, inputs=[model_file, model_folder], outputs=[chat_model_state, chat_tokenizer_state, load_status, chatbot])
                chat_input.submit(fn=_chat_fn, inputs=[chat_input, chatbot, chat_model_state, chat_tokenizer_state, temperature_slider, max_context_turns], outputs=[chatbot, chat_input])
        # ---- Performance Monitor ----
        monitor_panel = gr.HTML(monitor_html, every=1)
        gr.Markdown('<footer>AI Studio &mdash; PyTorch + Gradio</footer>')
    return demo

if __name__ == '__main__':
    import torch as _t
    print('='*50)
    print('  AI Studio - System Check')
    print('='*50)
    if _t.cuda.is_available():
        gpu = _t.cuda.get_device_name(0)
        vram = _t.cuda.get_device_properties(0).total_memory/(1024**3)
        print(f'  GPU: {gpu} ({vram:.1f} GB) | CUDA: {_t.version.cuda}')
    else:
        print('  WARNING: CUDA not available - running on CPU')
    print('='*50)
    print()
    demo = build_ui()
    demo.launch(server_name='127.0.0.1', server_port=int(os.environ.get('GRADIO_SERVER_PORT', 7860)), share=False, inbrowser=True, css=CUSTOM_CSS, theme=gr.themes.Soft())