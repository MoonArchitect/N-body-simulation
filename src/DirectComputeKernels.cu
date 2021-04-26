#include "DirectComputeKernels.cuh"

#define BLOCK_SIZE 128

void direct2DCompute(float4* pos, float4* acc, int n, const float SOFTENING) {
	int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	direct2DComputeKernel <<< nBlocks, BLOCK_SIZE >>> (pos, acc, n, SOFTENING);
}

__global__ void direct2DComputeKernel(float4* pos, float4* acc, int n, const float SOFTENING) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		float4 b = pos[i];
		float Fx = 0.0f; float Fy = 0.0f;

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
				float invDist3 = invDist * invDist * spos[j].w; //  * invDist
				Fx += dx * invDist3; Fy += dy * invDist3;
			}
			//__syncthreads();
		}

		//if (i != 0) {
		//	float dx = -b.x;
		//	float dy = -b.y;
		//	float r = sqrtf(SOFTENING + dx * dx + dy * dy);
		//	float mass = fmaxf(100000, r) * 1000;
		//	float invDist3 = 1.0f / (r * r * r) * mass;
		//	Fx += dx * invDist3; Fy += dy * invDist3;
		//}

		acc[i].x = Fx;
		acc[i].y = Fy;
	}
}



void direct3DCompute(float4* pos, float4* acc, int n, const float SOFTENING) {
	int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	direct3DComputeKernel <<< nBlocks, BLOCK_SIZE >>> (pos, acc, n, SOFTENING);
}

__global__ void direct3DComputeKernel(float4* pos, float4* acc, int n, const float SOFTENING) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		float4 b = pos[i];
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (unsigned int tile = 0; tile < gridDim.x; tile++) {
			__shared__ float4 spos[BLOCK_SIZE];
			spos[threadIdx.x] = pos[tile * blockDim.x + threadIdx.x];


			__syncthreads();

#pragma unroll
			for (int j = 0; j < BLOCK_SIZE; j++) {
				float dx = spos[j].x - b.x;
				float dy = spos[j].y - b.y;
				float dz = spos[j].z - b.z;
				float distSqr = SOFTENING + dx * dx + dy * dy + dz * dz;
				float invDist = rsqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist * spos[j].w;
				Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
			}
			//__syncthreads();
		}

		acc[i].x = Fx;
		acc[i].y = Fy;
		acc[i].z = Fz;
	}
}
