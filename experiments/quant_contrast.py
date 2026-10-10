"""Contrast: bf16 QK^T and W8A16 (K/V int8 storage) vs int8 QK^T under outliers."""
import math
import torch

DEVICE = "cuda"
torch.manual_seed(0)


def cosine_per_head(a, b):
    B, H, S, D = a.shape
    out = []
    for h in range(H):
        x = a[:, h].reshape(-1).double()
        y = b[:, h].reshape(-1).double()
        out.append((x @ y) / (x.norm() * y.norm() + 1e-12))
    return torch.stack(out)


def make_inputs(S, outlier_mode):
    B, H, D = 1, 8, 128
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    if outlier_mode == "uniform_rows":
        n = max(1, S // 100)
        idx = torch.randperm(S)[:n]
        f = torch.empty(n, device=DEVICE).uniform_(50, 5000)
        q[:, :, idx, :] *= f[None, None, :, None]
        k[:, :, idx, :] *= f[None, None, :, None]
    elif outlier_mode == "elements":
        n = max(1, S * D // 1000)
        for _ in range(n):
            s = torch.randint(0, S, (1,)).item()
            d = torch.randint(0, D, (1,)).item()
            q[:, :, s, d] *= float(torch.empty(1).uniform_(50, 5000))
    return q, k, v


def ref(q, k, v):
    D = q.shape[-1]
    a = q.float() @ k.float().transpose(-2, -1) * (1.0 / math.sqrt(D))
    return a.softmax(-1) @ v.float()


def bf16_qk(q, k, v):
    D = q.shape[-1]
    # bf16 storage, fp32 accumulate (exactly what my WMMA kernel does)
    a = q.float() @ k.float().transpose(-2, -1) * (1.0 / math.sqrt(D))
    return a.softmax(-1) @ v.float()


def w8a16(q, k, v):
    # K,V stored int8 (per-token scale), dequantized to bf16, QK^T and PV in bf16
    def quant_store(x):
        amax = x.abs().max(dim=-1, keepdim=True).values
        scale = amax / 127.0
        xq = torch.round(x.float() / scale).clamp(-127, 127) * scale
        return xq.to(torch.bfloat16)
    kk, vv = quant_store(k), quant_store(v)
    D = q.shape[-1]
    a = q.float() @ kk.float().transpose(-2, -1) * (1.0 / math.sqrt(D))
    return a.softmax(-1) @ vv.float()


for S in [10000, 24576]:
    for mode in ["none", "uniform_rows", "elements"]:
        q, k, v = make_inputs(S, mode)
        r = ref(q, k, v)
        cb = cosine_per_head(bf16_qk(q, k, v), r).min().item()
        cw = cosine_per_head(w8a16(q, k, v), r).min().item()
        print(f"S={S:6d} {mode:14s}  bf16_QK min_cos={cb:.6f} ({'PASS' if cb>=0.9999 else 'FAIL'})  "
              f"W8A16 min_cos={cw:.6f} ({'PASS' if cw>=0.9999 else 'FAIL'})", flush=True)
