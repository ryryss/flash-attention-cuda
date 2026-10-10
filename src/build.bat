@echo off
rem Build the custom FlashAttention CUDA kernel DLL for RTX 4070 SUPER (sm_89).
rem Run from anywhere; it cd's to this script's directory (src/) first.
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /MD -maxrregcount=128 -shared -o flash_attn.dll flash_attn_kernel.cu
if %ERRORLEVEL% EQU 0 (
    echo [OK] flash_attn.dll built.
) else (
    echo [FAIL] build error.
)
