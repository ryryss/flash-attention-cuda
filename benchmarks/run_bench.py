"""Correctness validation + benchmark for the custom FlashAttention kernel."""
import math
import os, sys
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flash_attn import flash_attn


def ref_fp32(q, k, v, causal):
    S, D = q.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    a = q.float() @ k.float().transpose(-2, -1) * scale
    if causal:
        mask = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), 1)
        a = a.masked_fill(mask, float("-inf"))
    return a.softmax(-1) @ v.float()


def check(B, H, S, D, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    mine = flash_attn(q, k, v, causal).float()
    ref = ref_fp32(q, k, v, causal)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        pt = F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()

    e_ref = (mine - ref).abs().max().item()
    e_pt = (mine - pt).abs().max().item()
    print(f"  B={B} H={H:2d} S={S:6d} D={D:3d} causal={int(causal)}   "
          f"max|mine-fp32ref|={e_ref:.3e}   max|mine-cuDNN|={e_pt:.3e}")
    return e_ref


def correctness():
    print("=" * 72)
    print("正确性校验 (my kernel vs fp32 参考 & vs PyTorch cuDNN)")
    print("=" * 72)
    worst = 0.0
    # S=500/1000 非 64 的倍数, 顺带测 padding
    for (B, H, S, D) in [(1, 1, 500, 128), (1, 2, 512, 128),
                         (1, 4, 1024, 128), (2, 4, 1000, 128),
                         (1, 8, 4096, 128)]:
        for causal in (False, True):
            worst = max(worst, check(B, H, S, D, causal))
    print(f"\n  worst max|mine-fp32ref| = {worst:.3e}  (bf16 精度 ~1e-2 属正常)")
    print()


def time_it(fn, warmup=5, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    t = torch.tensor(ts)
    return t.median().item(), t.mean().item(), t.min().item()


def benchmark():
    B, H, S, D = 1, 48, 36864, 128
    flops = 4 * B * H * S * S * D
    print("=" * 72)
    print(f"满规模基准  [B={B}, H={H}, S={S}, D={D}]  bf16")
    print("=" * 72)
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def tf(ms):
        return flops / 1e12 / (ms / 1e3)

    for causal in (False, True):
        tag = "causal" if causal else "non-causal"
        print(f"  [{tag}]")
        f = 0.5 * flops if causal else flops

        med, mean, mn = time_it(lambda: flash_attn(q, k, v, causal))
        print(f"    my-kernel     : median={med:8.2f} ms  mean={mean:8.2f}  min={mn:8.2f}  -> {f/1e12/(med/1e3):7.1f} TFLOPS")

        def pt_cudnn():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        med, mean, mn = time_it(pt_cudnn)
        print(f"    torch cuDNN   : median={med:8.2f} ms  mean={mean:8.2f}  min={mn:8.2f}  -> {f/1e12/(med/1e3):7.1f} TFLOPS")

        def pt_me():
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        med, mean, mn = time_it(pt_me)
        print(f"    torch mem_eff : median={med:8.2f} ms  mean={mean:8.2f}  min={mn:8.2f}  -> {f/1e12/(med/1e3):7.1f} TFLOPS")
        print()


if __name__ == "__main__":
    correctness()
    benchmark()
