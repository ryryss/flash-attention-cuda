# flash-attention-cuda

A from-scratch FlashAttention-style SDPA (Scaled Dot-Product Attention) CUDA kernel.
No attention library is used (not cutlass / xformers / official flash-attention):
hand-written CUDA with WMMA bf16 tensor cores, tuned step by step on an RTX 4070 SUPER.

- Exact attention `softmax(QK^T/√d)·V` with causal / non-causal.
- FlashAttention algorithm: tiling + online softmax, no S×S materialization.
- bf16 tensor cores (WMMA) + fp32 accumulation, register-resident S/P/acc.
- Shared-memory padding to kill bank conflicts; cp.async prefetch; 2 blocks/SM occupancy.

Target shape `[B=1, H=48, S=36864, D=128]` bf16 (measured on the same RTX 4070 SUPER):
**~820 ms** (40.7 TFLOPS) non-causal, roughly on par with PyTorch's mem_efficient
backend (~820–856 ms, within thermal noise); cuDNN is clearly faster (~565 ms).

> 中文说明见 [README_zh.md](./README_zh.md) · GPU terminology cheat-sheet: [GPU_Terminology_zh.md](./GPU_Terminology_zh.md)

## Structure

```
src/          # core: kernel, ctypes wrapper, build script
benchmarks/   # run_bench.py + ncu profiling targets
tests/        # correctness / debug scripts
experiments/  # layout dumps, quantization feasibility, v5 backup
```

## Build

```bat
src\build.bat
```

produces `src/flash_attn.dll`.

## Usage

```python
import os, sys
sys.path.insert(0, "path/to/src")   # make flash_attn importable
import torch
from flash_attn import flash_attn

q = torch.randn(1, 48, 36864, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
out = flash_attn(q, k, v, causal=False)
```

## Limitations

- D=128, bf16, single GPU only.
