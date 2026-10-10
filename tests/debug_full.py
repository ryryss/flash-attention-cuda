import os, sys
import torch, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from flash_attn import flash_attn

B, H, S, D = 1, 48, 36864, 128
torch.manual_seed(0)
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)

for causal in (False, True):
    # warmup
    flash_attn(q, k, v, causal); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record(); flash_attn(q, k, v, causal); e.record(); torch.cuda.synchronize()
    print(f"my-kernel S={S} causal={causal}: {s.elapsed_time(e):.1f} ms", flush=True)
