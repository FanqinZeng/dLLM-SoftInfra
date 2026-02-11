import transformers

from omegaconf import DictConfig
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel


def load_pretrained_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    """
    Load a pretrained model based on the configuration.
    """
    from ..models import LLaDAModelLM, DreamModel, DParallelLLaDAModel, DParallelDreamModel, D2FLLaDAModel, D2FDreamModel

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        return LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dream":
        return DreamModel.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dparallel_llada":
        return DParallelLLaDAModel.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dparallel_dream":
        return DParallelDreamModel.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "fast_dllm_v2_1.5b" or model_family == "fast_dllm_v2_7b":
        from ..models import FastdLLMv2Model
        return FastdLLMv2Model.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "d2f_llada":
        print(f"Loading Base Model from: {cfg.model.path}")
        base_model = D2FLLaDAModel.from_pretrained(cfg.model.path, **model_kwargs)
        if hasattr(cfg.model, "lora_path") and cfg.model.lora_path:
            print(f"Loading D2F LoRA Adapter from: {cfg.model.lora_path}")
            model = PeftModel.from_pretrained(base_model, cfg.model.lora_path)
            model = model.merge_and_unload()
            return model
        else:
            raise ValueError("Config for 'd2f_llada' requires a valid path pointing to the LoRA weights.")
    elif model_family == "d2f_dream":
        print(f"Loading Base Model from: {cfg.model.path}")
        base_model = D2FDreamModel.from_pretrained(cfg.model.path, **model_kwargs)
        if hasattr(cfg.model, "lora_path") and cfg.model.lora_path:
            print(f"Loading D2F LoRA Adapter from: {cfg.model.lora_path}")
            model = PeftModel.from_pretrained(base_model, cfg.model.lora_path)
            model = model.merge_and_unload()
            return model
        else:
            raise ValueError("Config for 'd2f_dream' requires a valid path pointing to the LoRA weights.")

    raise ValueError(f"Unsupported pretrained model: {cfg.model.name}")


def load_eval_model(cfg: DictConfig, **model_kwargs):
    from ..models import LLaDAEval, DreamEval, DParallelLLaDAEval, DParallelDreamEval, D2FLLaDAEval, D2FDreamEval, FastdLLMv2Eval

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == "dream":
        eval_model = DreamEval(cfg, **model_kwargs)
    elif model_family == "dparallel_llada":
        eval_model = DParallelLLaDAEval(cfg, **model_kwargs)
        print(model_family)
    elif model_family == "dparallel_dream":
        eval_model = DParallelDreamEval(cfg, **model_kwargs)
    elif model_family == 'd2f_llada':
        eval_model = D2FLLaDAEval(cfg, **model_kwargs)
    elif model_family == 'd2f_dream':
        eval_model = D2FDreamEval(cfg, **model_kwargs)
    elif model_family == "fast_dllm_v2_1.5b" or model_family == "fast_dllm_v2_7b":
        eval_model = FastdLLMv2Eval(cfg, **model_kwargs)
    else:
        raise NotImplementedError(
            f"Model family {model_family} is not implemented for evaluation."
        )

    return eval_model


def load_tokenizer(cfg: DictConfig, **tokenizer_kwargs):

    # ---------------- Tokenizer loading ----------------
    tokenizer_kwargs["trust_remote_code"] = True
    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada" or model_family == "dream":
        tokenizer=transformers.AutoTokenizer.from_pretrained(cfg.model.path,**tokenizer_kwargs)
    elif model_family == "fast_dllm_v2_1.5b" or model_family == "fast_dllm_v2_7b":
        print("Loading tokenizer from {cfg.model.tokenizer_path}")
        tokenizer=transformers.AutoTokenizer.from_pretrained(cfg.model.tokenizer_path,**tokenizer_kwargs)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Model-specific customization ----------------
    match model_family:
        case "llada" | "dparallel_llada" | 'd2f_llada':
            tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
            tokenizer.eot_token = "<|eot_id|>"

            # fix bugs in chat template
            tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
        case "dream" | "dparallel_dream" | 'd2f_dream':
            tokenizer.eot_token = "<|im_end|>"
            tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.eot_token
            )
        case "fast_dllm_v2_1.5b" | "fast_dllm_v2_7b":
            tokenizer.bos_token = "<|endoftext|>"
            tokenizer.eot_token = "<|im_end|>"
            tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.eot_token
            )
    return tokenizer
