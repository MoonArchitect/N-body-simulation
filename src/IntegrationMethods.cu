//#include "IntegrationMethods.cuh"
#include "NbodySystem.h"
#include "DirectComputeKernels.cuh"

#define BLOCK_SIZE 128

using namespace IntegrationMethods;

void IntegrationMethod::setSystem(NbodySystem* system) {
	this->system = system;
}


__global__ void R2EulerIntegration(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * acc[i].x; 
		vel[i].y += dt * acc[i].y;

		pos[i].x += vel[i].x * dt; 
		pos[i].y += vel[i].y * dt;
	}
}

__global__ void R3EulerIntegration(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * acc[i].x;
		vel[i].y += dt * acc[i].y;
		vel[i].z += dt * acc[i].z;

		pos[i].x += vel[i].x * dt;
		pos[i].y += vel[i].y * dt;
		pos[i].z += vel[i].z * dt;
	}
}

void Euler::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	system->computeAcceleration();
	if(system->space == R2)
		R2EulerIntegration <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	else
		R3EulerIntegration <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	cudaDeviceSynchronize();
}
