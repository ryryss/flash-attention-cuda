"""Feasibility check: can int8 QK^T meet cosine>=0.9999 (vs bf16 SDPA), incl. outliers?"""
import math
import torch
import torch.nn.functional as F

DEVICE = "cuda"
torch.manual_seed(0)


def ref_bf16(q, k, v):
    S, D = q.shape[-2], q.shape[-1]
    a = q.float() @ k.float().transpose(-2, -1) * (1.0 / math.sqrt(D))
    p = a.softmax(-1)
    return p @ v.float()


def qk_int8(q, k, v, per_token=True):
    """simulate per-token int8 QK^T (dequant-then-float-matmul; int32 acc is exact, so this is a lower bound)"""
    def quant(x):
        amax = x.abs().max(dim=-1, keepdim=True).values
        scale = amax / 127.0
        xq = torch.round(x.float() / scale).clamp(-127, 127)
        return xq * scale
    qq, kk = quant(q), quant(k)
    D = q.shape[-1]
    a = qq @ kk.transpose(-2, -1) * (1.0 / math.sqrt(D))
    p = a.softmax(-1)
    return p @ v.float()


def cosine_per_head(a, b):
    B, H, S, D = a.shape
    out = []
    for h in range(H):
        x = a[:, h].reshape(-1).double()
        y = b[:, h].reshape(-1).double()
        out.append((x @ y) / (x.norm() * y.norm() + 1e-12))
    return torch.stack(out)


def run(S, outlier_mode):
    B, H, D = 1, 8, 128
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)

    if outlier_mode == "uniform_rows":
        # whole rows of Q scaled by 50..5000x
        n = max(1, S // 100)
        idx = torch.randperm(S)[:n]
        f = torch.empty(n, device=DEVICE).uniform_(50, 5000)
        q[:, :, idx, :] *= f[None, None, :, None]
    elif outlier_mode == "elements":
        # a few random elements of Q scaled by 50..5000x (breaks per-token scale)
        n = max(1, S * D // 1000)
        for _ in range(n):
            s = torch.randint(0, S, (1,)).item()
            d = torch.randint(0, D, (1,)).item()
            q[:, :, s, d] *= float(torch.empty(1).uniform_(50, 5000))

    ref = ref_bf16(q, k, v)
    mine = qk_int8(q, k, v)
    cos = cosine_per_head(mine, ref)
    return cos.min().item(), cos.mean().item()


for S in [4096, 10000]:
    for mode in ["none", "uniform_rows", "elements"]:
        mn, mean = run(S, mode)
        flag = "PASS" if mn >= 0.9999 else "FAIL"
        print(f"S={S:6d} outliers={mode:14s}  min_cosine={mn:.6f}  mean={mean:.6f}  [{flag}]", flush=True)
