import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from flash_attn import flash_attn

B, H, S, D = 1, 48, 36864, 128
torch.manual_seed(0)
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)

flash_attn(q, k, v, causal=False)   # warmup
torch.cuda.synchronize()
flash_attn(q, k, v, causal=False)
torch.cuda.synchronize()
print("done", flush=True)
