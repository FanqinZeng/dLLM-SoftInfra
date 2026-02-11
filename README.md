# d2Cache

d2Cache is a framework for efficient inference and caching in Diffusion Language Models (MDLMs). This repository contains several advanced model implementations and optimized inference strategies.

## 🚀 New Models and Usage

We have introduced several new model variants and optimized parallel versions. To use these models, you must configure the required environment variables and use the specific `model` parameter in `eval.py`.

### 1. Model Summary & Parameters

| Model Category | `model` Argument | Description | Required Environment Variables |
| :--- | :--- | :--- | :--- |
| **DParallel** | `dparallel_llada-inst` | Parallel Optimized LLaDA | `LLADA_INST_PATH` |
| | `dparallel_dream-inst` | Parallel Optimized Dream | `DREAM_BASE_PATH` |
| **D2F (LoRA)** | `d2f_llada-inst` | LLaDA with D2F LoRA | `LLADA_INST_PATH`, `D2F_LLADA_INST_PATH` |
| | `d2f_dream-inst` | Dream with D2F LoRA | `DREAM_BASE_PATH`, `D2F_DREAM_INST_PATH` |
| **Fast dLLM v2** | `fast_dllm_v2_1.5b-inst` | Optimized 1.5B Variant | `FAST_DLLM_V2_1_5_PATH`, `DREAM_BASE_PATH` (Tokenizer) |
| | `fast_dllm_v2_7b-inst` | Optimized 7B Variant | `FAST_DLLM_V2_7_PATH`, `DREAM_BASE_PATH` (Tokenizer) |

---

### 2. Environment Variables Configuration

Set the following variables based on the model you intend to use:

```bash
# === Base Model Paths ===
export LLADA_INST_PATH=/path/to/LLaDA-8B-Instruct
export DREAM_BASE_PATH=/path/to/Dream-Base-Model

# === D2F LoRA Adapter Paths (Required for D2F models) ===
export D2F_LLADA_INST_PATH=/path/to/d2f-llada-lora
export D2F_DREAM_INST_PATH=/path/to/d2f-dream-lora

# === Fast dLLM v2 Model Paths (Required for Fast dLLM models) ===
export FAST_DLLM_V2_1_5_PATH=/path/to/fast-dllm-v2-1.5b
export FAST_DLLM_V2_7_PATH=/path/to/fast-dllm-v2-7b
```

---

### 3. Quick Start Commands

#### A. Running DParallel LLaDA
```bash
export LLADA_INST_PATH=/path/to/LLaDA-8B-Instruct

accelerate launch eval.py \
    model=dparallel_llada-inst \
    dataset.name=gsm8k \
    batch_size=1
```

#### B. Running D2F LLaDA (with Base Model and LoRA)
```bash
export LLADA_INST_PATH=/path/to/LLaDA-8B-Instruct
export D2F_LLADA_INST_PATH=/path/to/d2f-llada-lora

accelerate launch eval.py \
    model=d2f_llada-inst \
    dataset.name=gsm8k \
    batch_size=1
```

#### C. Running Fast dLLM v2 7B (with Tokenizer Path)
```bash
export FAST_DLLM_V2_7_PATH=/path/to/fast-dllm-v2-7b
export DREAM_BASE_PATH=/path/to/Dream-Base-Model  # Required for tokenizer

accelerate launch eval.py \
    model=fast_dllm_v2_7b-inst \
    dataset.name=gsm8k \
    batch_size=1
```

---

## 🛠 Advanced Configuration

All configurations are managed via [Hydra](https://hydra.cc/). You can override any parameters from the command line:

- **Cache Strategy**: `cache=prefix` (options: `prefix`, `d2cache`, `dllm`)
- **Generation Strategy**: `generation=vanilla` (options: `vanilla`, `eb_sampler`, `pc_sampler`, `klass`, `wino`, `daedal`)
- **Evaluation Limit**: `eval_args.limit=10`

For more detailed information on adding new models, please refer to [AddModel.md](AddModel.md).

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
