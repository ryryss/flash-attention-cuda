// Dump WMMA bf16 matrix_a fragment layout (16x16x16) with exact encoding row*16+col.
#include <cstdio>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
using bf16 = __nv_bfloat16;

__global__ void dump_a_kernel(int* outA) {
    __shared__ bf16 M[16 * 16];
    for (int i = threadIdx.x; i < 16 * 16; i += 32)
        M[i] = __float2bfloat16((float)((i / 16) * 16 + (i % 16)));  // row*16+col, exact in bf16

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa;
    wmma::load_matrix_sync(fa, M, 16);

    int t = threadIdx.x;
    for (int i = 0; i < 8; i++) outA[t * 8 + i] = (int)__bfloat162float(fa.x[i]);
}

int main() {
    int *dA, hA[32*8];
    cudaMalloc(&dA, 32*8*4);
    dump_a_kernel<<<1, 32>>>(dA);
    cudaMemcpy(hA, dA, 32*8*4, cudaMemcpyDeviceToHost);
    printf("matrix_a row_major: (row,col) = (v/16, v%%16)\n");
    for (int t = 0; t < 32; t++) {
        printf("t=%2d: ", t);
        for (int i = 0; i < 8; i++) printf("(%d,%d) ", hA[t*8+i]/16, hA[t*8+i]%16);
        printf("\n");
    }
    cudaFree(dA);
    return 0;
}
