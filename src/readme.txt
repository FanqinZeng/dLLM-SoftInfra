d2Cache Source Directory
======================

This directory contains the core implementation of the d2Cache framework, designed for efficient inference and caching in Diffusion Language Models (MDLMs) such as LLaDA and Dream.

Core Modules:
-------------

1. cache/
   Implements various KV cache management strategies:
   - base.py: Base dCache class and context managers.
   - d2cache.py: Dynamic cache updating using Attention Rollout.
   - dllm_cache.py: Adaptive refreshing for prompt/response sequences.
   - prefix_cache.py: Support for semi-autoregressive remasking.

2. generation/
   Advanced decoding strategies for diffusion models:
   - vanilla.py: Standard denoising generation.
   - daedal.py: Dynamic generation length adjustment.
   - klass.py: KL-Adaptive Stability Sampling.
   - wino.py: Wino-based constrained decoding.

3. models/
   Custom model implementations and evaluation wrappers:
   - llada/: Support for LLaDA models.
   - dream/: Support for Dream models.
   - variants: dparallel (parallel optimization) and d2f (LoRA support).
   - eval_mdlm.py: Base class for MDLM evaluation.

4. frame.py
   Defines data structures for tracking decoding states:
   - Frame: State of a single decoding step.
   - FrameDelta: Incremental changes between steps.
   - DecodeRecord: History of the entire generation process.

5. utils/
   Shared utility functions:
   - common.py: Timer, Registry, and specialized tensor operations (insert/delete).
   - models.py: Utilities for loading pretrained models and tokenizers.
   - third_party/: External resources like token frequency corpora.

Usage:
------
The framework is designed to be used by registering generation strategies and injecting cache objects into model forward passes via context managers.
