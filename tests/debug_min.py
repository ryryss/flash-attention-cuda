import os, sys
import torch
print("step1: import torch ok", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flash_attn import flash_attn
print("step2: import flash_attn ok", flush=True)

torch.manual_seed(0)
q = torch.randn(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
print("step3: tensors ok", flush=True)

print("step4: launching kernel...", flush=True)
o = flash_attn(q, k, v, causal=False)
print("step5: kernel launched, syncing...", flush=True)
torch.cuda.synchronize()
print("step6: sync ok, o.shape=", tuple(o.shape), flush=True)

print("o[0,0,0,:4]=", o[0, 0, 0, :4].float().tolist(), flush=True)
print("DONE", flush=True)
