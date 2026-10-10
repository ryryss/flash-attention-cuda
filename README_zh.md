# flash-attention-cuda

从零实现的 FlashAttention 风格 SDPA（Scaled Dot-Product Attention）CUDA kernel。
不使用任何现成 attention 库（非 cutlass / 非 xformers / 非官方 flash-attention），
纯 CUDA + WMMA 张量核手写，并针对 RTX 4070 SUPER 逐级调优。

## 实现了什么功能

- **精确注意力**：`softmax(QKᵀ/√d)·V`，支持 causal（因果 mask）与 non-causal（全注意力）。
- **FlashAttention 核心算法**：分块 tiling + 在线 softmax（online softmax），
  全程不实化 S×S 注意力矩阵（S=36864 时该矩阵 fp32 达 243GB，必须分块）。
- **bf16 张量核计算**：QKᵀ 与 PV 两个矩阵乘用 `nvcuda::wmma` bf16 tensor core 指令，fp32 累加。
- **寄存器驻留**：S、P、累加器 acc 全程放在 WMMA fragment 寄存器里，shared 只放 Q/K/V。
- **共享内存 padding**：stride 128→136 消除 8 路 bank conflict（实测提速 2 倍）。
- **cp.async 异步预取**：K/V 用 `__pipeline_memcpy_async` 重叠内存延迟。
- **occupancy 调优**：通过压寄存器（`-maxrregcount=128`）+ 单缓冲 K/V，达到 2 blocks/SM（16 warps）。

## 本机硬件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4070 SUPER（12GB，计算能力 sm_89） |
| CUDA | 12.6 runtime（nvcc 12.4 编译） |
| cuDNN | 9.10 |
| PyTorch | 2.14.1+cu126 |
| 操作系统 | Windows |
| 编译器 | MSVC 2019 + nvcc |

## 实测性能（目标形状 [B=1, H=48, S=36864, D=128] bf16）

| 实现 | non-causal | causal |
|------|-----------|--------|
| **本 kernel** | **820 ms（40.7 TFLOPS）** | 839 ms |
| torch cuDNN（闭源） | 565 ms（59.1 TFLOPS） | 295 ms |
| torch mem_efficient（开源） | 856 ms（39.0 TFLOPS） | 433 ms |

- 非因果下**反超 PyTorch mem_efficient**，达 cuDNN 的 0.69x。
- 正确性：vs fp32 参考实现，max abs err ≈ 8e-3（bf16 精度内）。

## 迭代历程（累计 13.4 倍加速）

| 版本 | 非因果耗时 | 关键优化 |
|------|-----------|---------|
| v1 共享内存版 | ~11000 ms | 首版 |
| v2 寄存器驻留 S/P/acc | ~4800 ms | WMMA fragment 寄存器 |
| v3 BLOCK_M=128 | ~3670 ms | K/V 读取减半 |
| v4 cp.async 双缓冲 | ~1690 ms | 异步预取 |
| v5 padding 消 bank conflict | ~880 ms | conflict 归零（2x）|
| v6 2 blocks/SM | ~820 ms | occupancy 16.67%→33.33% |

## 目录结构

```
flash-attention-cuda/
├── src/                        # 核心实现
│   ├── flash_attn_kernel.cu    # CUDA kernel（当前 v6）
│   ├── flash_attn.py           # ctypes 封装
│   └── build.bat               # 一键编译
├── benchmarks/                 # 基准测试
│   ├── run_bench.py            # 正确性 + 满规模基准
│   ├── profile_target.py       # ncu profiling（小规模）
│   └── profile_full.py         # ncu profiling（满规模）
├── tests/                      # 正确性/调试测试
│   ├── debug_min.py
│   ├── debug_shapes.py
│   └── debug_full.py
├── experiments/                # 实验与探索
│   ├── dump_*.cu               # WMMA fragment 布局实测
│   ├── quant_*.py              # int8 量化可行性验证
│   └── flash_attn_kernel_v5_backup.cu   # v5 备份
├── README.md                   # 英文简介
├── README_zh.md                # 中文说明（本文档）
└── GPU_Terminology_zh.md       # GPU 术语速查
```

## 编译

```bat
src\build.bat
```

编译产物 `flash_attn.dll` 生成在 `src/` 下。

## 使用

```python
import os, sys
sys.path.insert(0, "path/to/src")   # 让 flash_attn 可导入
import torch
from flash_attn import flash_attn

q = torch.randn(1, 48, 36864, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
out = flash_attn(q, k, v, causal=False)
```

## 已知局限

- 仅支持 D=128、bf16、单 GPU。
- 当前瓶颈为串行结构 QK^T→softmax→PV 造成的张量核空转（利用率 ~54%），
  下一步优化方向为 software pipelining（交错相邻 tile 的计算填满空档）。
