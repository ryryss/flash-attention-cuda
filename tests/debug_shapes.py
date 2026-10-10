import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from flash_attn import flash_attn

for (B, H, S, D) in [(1,1,500,128),(1,2,512,128),(1,4,1024,128),(2,4,1000,128),(1,8,4096,128)]:
    for causal in (False, True):
        torch.manual_seed(0)
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        print(f"launch B={B} H={H} S={S} causal={causal} ...", flush=True)
        o = flash_attn(q, k, v, causal)
        torch.cuda.synchronize()
        print(f"  ok  o[0,0,0,:3]={o[0,0,0,:3].float().tolist()}", flush=True)
print("ALL DONE", flush=True)
