# 报错原因分析：`ValueError: Token frequency not initialized for debiasing.`

## 1. 问题触发点
当你在执行命令时，配置中包含了：
```yaml
generation=pc_sampler
generation.debias=true
```
由于指定了 `generation.debias=true`，程序在生成文本时会启用 token 去偏（debiasing）功能。

## 2. 代码执行链路与错误位置
报错堆栈的末端位于 `src/generation/utils.py` 的 `sample_tokens` 函数中（约第 114-115 行）：
```python
    if debias:
        global _token_freq
        alpha = clip_alpha if clip_alpha is not None else 10.0
        if _token_freq is None:
            raise ValueError("Token frequency not initialized for debiasing.")
```
这里逻辑表明：当 `debias` 开启时，算法需要一个预先加载好的 `_token_freq`（词频分布数据）来进行计算。如果这个全局变量没被初始化，就会直接抛出该 `ValueError` 异常。

## 3. 导致未初始化的根本原因
通过全项目检索，找到了加载 `_token_freq` 数据的方法：作者在 `src/third_party/__init__.py` 中实现了 `get_token_freq(model)` 函数，这个函数会从相应的 json 文件（如 `llada_corpus.json`）读取频率并生成 Tensor。

**但是（Bug 所在）**：
项目在生成流程的主入口（如 `src/generation/__init__.py` 中的 `generate` 函数，或者是 `src/generation/vanilla.py` 中的 `vanilla_generate` 函数）里，**均没有任何代码去调用 `get_token_freq()` 函数并对 `src.generation.utils._token_freq` 进行赋值初始化**。

这属于**代码遗漏的 Bug**：作者虽然写了加载频率的功能，也写了如果未加载就报错的检查，但在将它们串联起来的生成入口处，忘记加上“初始化频率”的逻辑代码了。导致只要用户开启 `debias=true`，程序就百分之百会由于找不到初始化的频率数据而崩溃。

## 4. 后续修复建议（目前未做修改）
如果之后需要修复此 Bug，可以在 `src/generation/__init__.py` 的 `generate` 函数内增加拦截。当发现传入了 `debias=True` 参数时，先主动调用 `get_token_freq` 完成全局变量初始化：
```python
        if kwargs.get("debias", False):
            import src.generation.utils as gen_utils
            if gen_utils._token_freq is None:
                from src.third_party import get_token_freq
                gen_utils._token_freq = get_token_freq(model, device=model.device, dtype=model.dtype)
```