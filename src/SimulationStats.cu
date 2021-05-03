#include <tuple>
#include "NbodySystem.h"

#define BLOCK_SIZE 128


std::tuple<double, double, double> SimulationStats::computeLinearMomentum(Space space, float4* pos, float4* vel, int n) {
	double* dm; cudaMalloc(&dm, sizeof(double)); cudaMemset(dm, 0, sizeof(double));
	double* dkE; cudaMalloc(&dkE, sizeof(double)); cudaMemset(dkE, 0, sizeof(double));
	
	if(space == R2)
		SimulationStats::compute_LMoment_kE_R2Kernel << < 64, BLOCK_SIZE >> > (dm, dkE, pos, vel, n);
	else
		SimulationStats::compute_LMoment_kE_R3Kernel << < 64, BLOCK_SIZE >> > (dm, dkE, pos, vel, n);
	cudaDeviceSynchronize();

	double lmoment; cudaMemcpy(&lmoment, dm, sizeof(double), cudaMemcpyDeviceToHost);
	double kE; cudaMemcpy(&kE, dkE, sizeof(double), cudaMemcpyDeviceToHost);
	return { lmoment, kE, -1 };
}

__global__ void SimulationStats::compute_LMoment_kE_R2Kernel(double* momentum, double* kE, float4* pos, float4* vel, int n) {
	int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int stride = BLOCK_SIZE * gridDim.x;
	double lm = 0, lkE = 0;

	while (i < n) {
		double v = sqrt(vel[i].x * vel[i].x + vel[i].y * vel[i].y);
		lm += v * pos[i].w;
		lkE += 0.5 * v * v * pos[i].w;

		i += stride;
	}

	atomicAdd(momentum, lm);
	atomicAdd(kE, lkE);
}

__global__ void SimulationStats::compute_LMoment_kE_R3Kernel(double* momentum, double* kE, float4* pos, float4* vel, int n) {
	int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int stride = BLOCK_SIZE * gridDim.x;
	double lm = 0, lkE = 0;

	while (i < n) {
		double v = sqrt(vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z);
		lm += v * pos[i].w;
		lkE += 0.5 * v * v * pos[i].w;

		i += stride;
	}

	atomicAdd(momentum, lm);
	atomicAdd(kE, lkE);
}


__global__ void SimulationStats::compute_pE_Kernel(double* pE, float4* pos, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		float4 b = pos[i];
		double lpE = 0;

		for (unsigned int tile = 0; tile < gridDim.x; tile++) {
			__shared__ float4 spos[BLOCK_SIZE];
			spos[threadIdx.x] = pos[tile * blockDim.x + threadIdx.x];

			__syncthreads();

#pragma unroll
			for (int j = 0; j < BLOCK_SIZE; j++) {
				float dx = spos[j].x - b.x;
				float dy = spos[j].y - b.y;
				float distSqr = dx * dx + dy * dy;
				float invDist = rsqrtf(distSqr);
				float invDist3 = invDist * invDist * spos[j].w;
			}
		}

		atomicAdd(pE, lpE);
	}
}
