#include "DirectComputeKernels.cuh"

#define BLOCK_SIZE 128

void directCompute(float4* pos, float3* acc, int n, const float SOFTENING) {
	int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	directComputeKernel << < nBlocks, BLOCK_SIZE >> > (pos, acc, n, SOFTENING);
}

// 2D
__global__ void directComputeKernel(float4* pos, float3* acc, int n, const float SOFTENING) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		float4 b = pos[i];
		float Fx = 0.0f; float Fy = 0.0f;// float Fz = 0.0f;

		for (unsigned int tile = 0; tile < gridDim.x; tile++) {
			__shared__ float4 spos[BLOCK_SIZE];
			spos[threadIdx.x] = pos[tile * blockDim.x + threadIdx.x];


			__syncthreads();

#pragma unroll
			for (int j = 0; j < BLOCK_SIZE; j++) {
				float dx = spos[j].x - b.x;
				float dy = spos[j].y - b.y;
				float distSqr = SOFTENING + dx * dx + dy * dy;
				float invDist = rsqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist * b.w;
				Fx += dx * invDist3; Fy += dy * invDist3;
			}
			__syncthreads();
		}

		acc[i].x = Fx;
		acc[i].y = Fy;
	}
}

