#include "NbodySystem.h"
#include "DirectComputeKernels.cuh"
#include "BarnesHutKernels.cuh"

using namespace ComputeMethods;


ComputeMethod::ComputeMethod(int knodes, const float softening) : knodes(knodes), SOFTENING(softening) {}




Direct::Direct(const float softening) : ComputeMethod(1, softening) {}

void Direct::setSystem(NbodySystem* system) {
	this->system = system;
}

void Direct::computeAcc() {
	if(system->space == R2)
		direct2DCompute(system->device.pos_mass, system->device.acc, system->N, SOFTENING);
	else
		direct3DCompute(system->device.pos_mass, system->device.acc, system->N, SOFTENING);
}




BarnesHut::BarnesHut(float theta, int knodes, const float softening)
	: theta(theta), ComputeMethod(knodes, softening) {}

void BarnesHut::setSystem(NbodySystem* system) {
	this->system = system;

	if (system->space == R3){
		printf("\n\n\nBarnes-Hut algorithm is not yet available for simulations in 3D space\nSimulating in 2D ...\n\n");
		system->space = R2;
	}

	cudaMalloc(&d_bounds, sizeof(float4));
	cudaMalloc(&d_index, sizeof(int));
	cudaMalloc(&d_nodes, 4 * system->M * sizeof(int));
	cudaMalloc(&sortedIdx, system->N * sizeof(int));
	cudaMalloc(&SFCkeys, system->N * sizeof(int));
}

void BarnesHut::computeAcc() {
	barnesHutCompute(
		system->device.pos_mass, system->device.acc, d_bounds, 
		d_index, d_nodes, sortedIdx, SFCkeys,
		system->N, system->M, theta, SOFTENING
	);
}

