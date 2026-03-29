# 🧠 Finetuning-LLM

> A practical, hands-on reference repository for fine-tuning large language models (LLMs) — covering techniques from full fine-tuning to parameter-efficient methods (LoRA, QLoRA), quantization, alignment strategies (SFT, RLHF, DPO), and the foundational Transformer architecture.

---

## 📁 Repository Contents

| File | Description |
|---|---|
| [`Fine_Tuning_LLm_Models.ipynb`](./Fine_Tuning_LLm_Models.ipynb) | General LLM fine-tuning walkthrough — concepts and starter code |
| [`Fine_Tuning_with_Mistral_QLora_PEFt.ipynb`](./Fine_Tuning_with_Mistral_QLora_PEFt.ipynb) | End-to-end fine-tuning of Mistral-7B using QLoRA + PEFT |
| [`Fine_tune_Llama_2.ipynb`](./Fine_tune_Llama_2.ipynb) | Fine-tuning LLaMA-2 with LoRA adapters |
| [`fine-tune-llama-3-1-step-by-step-guide.ipynb`](./fine-tune-llama-3-1-step-by-step-guide.ipynb) | Step-by-step guide for fine-tuning LLaMA 3.1 |
| [`lora_tuning.ipynb`](./lora_tuning.ipynb) | Deep-dive into LoRA adapter training |
| [`notes.md`](./notes.md) | 📓 Comprehensive reference notes with diagrams, formulas, and paper citations |

---

## 🔍 What This Repo Covers

### 1. Fine-Tuning Techniques
- **Full Fine-Tuning** — Update all model weights; highest accuracy, highest cost
- **Supervised Fine-Tuning (SFT)** — Train on prompt→response pairs; foundation of instruction following
- **RLHF** (Reinforcement Learning from Human Feedback) — Align models with human preferences via a reward model + PPO
- **DPO** (Direct Preference Optimization) — Simpler, more stable RLHF alternative with no reward model

### 2. Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA** — Inject trainable low-rank matrices `ΔW = BA` into frozen weights; trains <1% of parameters
- **QLoRA** — Combines 4-bit (NF4) quantization of the base model with LoRA adapters; enables 70B+ model fine-tuning on a single GPU
- Variants: DoRA, AdaLoRA, LoRA+, DyLoRA, LoReFT

### 3. Quantization
- **Precision formats** — FP32 → FP16 / BF16 → INT8 → INT4 → 1-bit
- **Symmetric vs Asymmetric quantization** — Formulas, use cases, tradeoffs
- **Post-Training Quantization (PTQ)** — GPTQ, AWQ, GGUF (llama.cpp), bitsandbytes
- **Quantization-Aware Training (QAT)** — BitNet b1.58 (ternary weights: {-1, 0, +1})

### 4. Transformer Architecture
- Scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`
- Multi-head attention, positional encoding (sinusoidal, RoPE, ALiBi)
- Encoder-only / Decoder-only / Encoder-Decoder variants
- Scaling from 65M (original paper) → 405B (LLaMA 3.1)

---

## 🚀 Quickstart: Fine-Tuning with QLoRA (Mistral-7B)

### Prerequisites

```bash
pip install transformers peft bitsandbytes trl accelerate datasets
```

### Key Libraries

| Library | Purpose |
|---|---|
| `transformers` | Load and run HuggingFace models |
| `peft` | LoRA / QLoRA adapter support |
| `bitsandbytes` | 4-bit & 8-bit quantization |
| `trl` | SFTTrainer, DPO training |
| `accelerate` | Multi-GPU / mixed-precision training |
| `datasets` | Dataset loading and preprocessing |

### Minimal QLoRA Config

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — optimal for neural net weights
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,   # Double quantization for extra memory savings
)

# LoRA adapter config
lora_config = LoraConfig(
    r=16,                # Rank — lower = fewer params, less expressive
    lora_alpha=32,       # Scaling factor (effective LR = alpha/r)
    target_modules=["q_proj", "v_proj"],  # Apply to attention Q and V matrices
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Training Pipeline

```
1. Load base model in 4-bit (QLoRA)
2. Attach LoRA adapters (trainable)
3. Prepare dataset in chat/instruction format
4. Train with SFTTrainer (TRL)
5. Merge adapters into base model
6. Export as GGUF / GPTQ for deployment
```

Refer to [`Fine_Tuning_with_Mistral_QLora_PEFt.ipynb`](./Fine_Tuning_with_Mistral_QLora_PEFt.ipynb) for the full end-to-end implementation.

---

## 🧮 Memory Requirements at a Glance

| Method | LLaMA-7B VRAM | LLaMA-70B VRAM |
|---|---|---|
| Full Fine-Tuning (FP16) | ~112 GB | ~1,120 GB |
| LoRA (FP16 base) | ~28 GB | ~140 GB |
| QLoRA (4-bit base) | ~5–6 GB | ~40–48 GB |

---

## 📚 Key Papers

| Paper | Authors | Year | Link |
|---|---|---|---|
| Attention Is All You Need | Vaswani et al. | 2017 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| LoRA: Low-Rank Adaptation of LLMs | Hu et al. | 2021 | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| InstructGPT (RLHF) | Ouyang et al. | 2022 | [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) |
| GPTQ | Frantar et al. | 2022 | [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) |
| LLaMA 2 | Touvron et al. | 2023 | [arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288) |
| QLoRA | Dettmers et al. | 2023 | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) |
| DPO | Rafailov et al. | 2023 | [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) |
| Mistral-7B | Jiang et al. | 2023 | [arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825) |
| BitNet b1.58 | Ma et al. | 2024 | [arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764) |
| DoRA | Liu et al. | 2024 | [arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353) |

---

## 🛠️ Useful Tools & Platforms

| Tool | Use |
|---|---|
| [HuggingFace Hub](https://huggingface.co) | Model hosting, datasets, PEFT/TRL libraries |
| [Unsloth](https://github.com/unslothai/unsloth) | 2× faster LoRA training, lower VRAM |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | Config-driven fine-tuning framework |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | UI + CLI for easy fine-tuning |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU inference with GGUF quantization |
| [Ollama](https://ollama.com) | Local LLM serving |
| [Together AI](https://www.together.ai) | Managed fine-tuning API |
| [Gradient.ai](https://gradient.ai) | Fine-tuning as a service |

---

## 📖 Notes

For in-depth explanations with Mermaid diagrams, math formulas, and paper breakdowns, see **[`notes.md`](./notes.md)**.

Topics covered in `notes.md`:
- Transformer architecture deep-dive (from *Attention Is All You Need*)
- LoRA matrix decomposition with worked examples
- Quantization math (symmetric / asymmetric formulas)
- RLHF sequence diagram and DPO loss function
- Decision flowchart: which technique to use based on available GPU

---

## 📄 License

This repository is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You are free to use, modify, and distribute this code under the terms of the GPL-3.0. Any derivative works must also be open-sourced under the same license.

See the full license text in [`LICENSE`](./LICENSE) or at [gnu.org/licenses/gpl-3.0](https://www.gnu.org/licenses/gpl-3.0.html).

---

## ⚠️ Disclaimer

- Model weights referenced or used in these notebooks (LLaMA, Mistral, etc.) are subject to their own respective licenses. Please review and accept the appropriate licenses on HuggingFace before use.
- Fine-tuning large models requires significant compute resources. Validate your hardware requirements using the memory table above before starting.
- Outputs from fine-tuned models should be evaluated carefully before deployment in production systems.

---

*Last updated: March 2026*
