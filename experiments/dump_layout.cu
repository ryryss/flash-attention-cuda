// Dump WMMA m16n16k16 f32 accumulator fragment layout.
#include <cstdio>
#include <mma.h>
using namespace nvcuda;

__global__ void dump_layout_kernel(float* out) {
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;
    // matrix C[i][j] = i * 1000 + j, row-major, ldm = 16
    __shared__ float C[16 * 16];
    for (int i = threadIdx.x; i < 16 * 16; i += 32) C[i] = (i / 16) * 1000.0f + (i % 16);

    C_frag f;
    wmma::load_matrix_sync(f, C, 16, wmma::mem_row_major);

    int t = threadIdx.x;
    // out[t*8 + i] = fragment value at index i for this thread
    for (int i = 0; i < 8; i++) out[t * 8 + i] = f.x[i];
}

int main() {
    float* d_out; float h_out[32 * 8];
    cudaMalloc(&d_out, 32 * 8 * sizeof(float));
    dump_layout_kernel<<<1, 32>>>(d_out);
    cudaMemcpy(h_out, d_out, 32 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("thread  val0..val7 -> (row=val/1000, col=val%%1000)\n");
    for (int t = 0; t < 32; t++) {
        printf("t=%2d: ", t);
        for (int i = 0; i < 8; i++) {
            int v = (int)h_out[t * 8 + i];
            printf("(%d,%d) ", v / 1000, v % 1000);
        }
        printf("\n");
    }
    cudaFree(d_out);
    return 0;
}
