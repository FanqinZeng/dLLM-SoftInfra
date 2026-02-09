# 如何添加新模型

在项目中添加新模型（例如 `dparallel_llada`）主要涉及以下四个方面：

1.  **模型实现**：在 `src/models/` 下创建新目录并实现模型类。
2.  **模型导出**：在 `src/models/__init__.py` 中公开新模型类。
3.  **模型注册**：在 `src/utils/models.py` 中注册加载逻辑。
4.  **配置文件**：在 `configs/model/` 中添加 YAML 配置文件。

---

## 1. 模型实现 (`src/models/`)

在 `src/models/` 目录下创建一个与模型名称对应的文件夹（例如 `src/models/new_model/`）。

在该文件夹中，你需要实现以下文件（参考 `src/models/dparallel_llada/`）：

*   **`__init__.py`**: 暴露模型类和配置类。
*   **`configuration_new_model.py`**: 定义模型配置类 `NewModelConfig`。
*   **`modeling_new_model.py`**: 定义模型架构类 `NewModelLM` (通常继承自 `PreTrainedModel`)。
*   **`eval_model.py`**: 定义评估包装类 `NewModelEval`。

## 2. 模型导出 (`src/models/__init__.py`)

在 `src/models/__init__.py` 中，添加一行以导出你的新模型类，以便其他模块可以导入它们。

**示例修改**：
```python
from .dream import DreamModel, DreamConfig, DreamEval
from .llada import LLaDAModelLM, LLaDAConfig, LLaDAEval
# [新增] 导出新模型
from .new_model import NewModelLM, NewModelConfig, NewModelEval
```

如果你的新模型只是对现有模型的简单封装或重命名（如 `dparallel_llada`），你可以这样写：
```python
from .dparallel_llada import LLaDAModelLM as DParallelLLaDAModel, LLaDAConfig as DParallelLLaDAConfig, LLaDAEval as DParallelLLaDAEval
```

## 3. 模型注册 (`src/utils/models.py`)

这是最关键的一步。你需要修改 `src/utils/models.py` 中的三个函数，以便系统能够通过配置名称识别并加载你的模型。

### 3.1 `load_pretrained_model`

在 `load_pretrained_model` 函数中添加一个新的 `elif` 分支，用于根据 `model_family` 加载你的模型类。

**示例修改**：
```python
def load_pretrained_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    # ... 引入模型类 ...
    from ..models import LLaDAModelLM, DreamModel, NewModelLM # [新增]

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        return LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dream":
        return DreamModel.from_pretrained(cfg.model.path, **model_kwargs)
    # [新增] 处理新模型
    elif model_family == "new_model":
        return NewModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    
    raise ValueError(f"Unsupported pretrained model: {cfg.model.name}")
```

### 3.2 `load_eval_model`

在 `load_eval_model` 函数中添加分支，返回对应的评估类。

**示例修改**：
```python
def load_eval_model(cfg: DictConfig, **model_kwargs):
    # ... 引入评估类 ...
    from ..models import LLaDAEval, DreamEval, NewModelEval # [新增]

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    # ... 其他模型 ...
    # [新增]
    elif model_family == "new_model":
        eval_model = NewModelEval(cfg, **model_kwargs)
    
    # ...
    return eval_model
```

### 3.3 `load_tokenizer`

如果你的模型需要特殊的 Token 处理（例如添加 `mask_token` 或设置 `eot_token`），在 `load_tokenizer` 函数中进行配置。

**示例修改**：
```python
def load_tokenizer(cfg: DictConfig, **tokenizer_kwargs):
    # ... 加载 tokenizer ...
    
    model_family = cfg.model.name.split("-")[0]
    match model_family:
        case "llada":
            # ...
        case "dream":
            # ...
        # [新增] 配置新模型的 tokenizer
        case "new_model":
            # 示例：添加自定义 token
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
            tokenizer.eot_token = "<|end|>"
            # 如果需要修复 chat_template 也可以在这里设置
    
    return tokenizer
```

## 4. 配置文件 (`configs/model/`)

最后，在 `configs/model/` 目录下添加 YAML 配置文件。文件名建议采用 `[model_family]-[variant].yaml` 的格式。

**示例文件** (`configs/model/new_model-inst.yaml`):
```yaml
name: new_model-inst  # 模型名称，前缀必须与 load_pretrained_model 中的判断一致
path: /path/to/weights/new_model  # 模型权重路径
# 其他特定配置...
```

