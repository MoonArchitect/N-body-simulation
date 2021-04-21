//#include "IntegrationMethods.cuh"
#include "NbodySystem.h"
#include "DirectComputeKernels.cuh"
#define BLOCK_SIZE 128


using namespace IntegrationMethods;

void IntegrationMethod::setSystem(NbodySystem* system) {
	this->system = system;
}



__global__ void EulerIntegration(float4* pos, float3* vel, float3* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * acc[i].x; 
		vel[i].y += dt * acc[i].y;

		pos[i].x += vel[i].x * dt; 
		pos[i].y += vel[i].y * dt;
	}
}

void Euler::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	system->computeAcceleration();
	//directComputeKernel << < nBlocks, BLOCK_SIZE >> > (system->d_pos_mass, system->d_acc, system->N, 0.5f);
	EulerIntegration << < nBlocks, BLOCK_SIZE >> > (system->d_pos_mass, system->d_vel, system->d_acc, dt, system->N);
	cudaDeviceSynchronize();
}



