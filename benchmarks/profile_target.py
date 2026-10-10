import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from flash_attn import flash_attn

# profile at a moderate size for speed
B, H, S, D = 1, 8, 4096, 128
torch.manual_seed(0)
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)

for _ in range(3):
    flash_attn(q, k, v, causal=False)
torch.cuda.synchronize()
print("done", flush=True)
