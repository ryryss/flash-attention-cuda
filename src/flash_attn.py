"""ctypes wrapper for the custom FlashAttention CUDA kernel."""
import ctypes
import os
import torch

# The DLL is built (see build.bat) next to this file.
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn.dll")
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.flash_attn_bf16.restype = ctypes.c_int
        _lib.flash_attn_bf16.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_int,     # B
            ctypes.c_int,     # H
            ctypes.c_int,     # S
            ctypes.c_int,     # D
            ctypes.c_int,     # causal
            ctypes.c_void_p,  # cudaStream_t
        ]
    return _lib


def flash_attn(q, k, v, causal=False):
    """Custom FlashAttention-style SDPA.  q,k,v: (B,H,S,D) bf16 CUDA, D must be 128."""
    assert q.is_cuda and q.dtype == torch.bfloat16, "q must be bf16 CUDA tensor"
    assert q.shape == k.shape == v.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    B, H, S, D = q.shape
    assert D == 128, "kernel currently targets D=128"

    o = torch.empty_like(q)
    stream = torch.cuda.current_stream().cuda_stream
    rc = _get_lib().flash_attn_bf16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
        B, H, S, D, int(causal), ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"flash_attn kernel returned error code {rc}")
    return o
