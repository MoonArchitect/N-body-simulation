//#include "ComputeMethods.h"
#include "NbodySystem.h"
#include "DirectComputeKernels.cuh"
#include "BarnesHutKernels.cuh"

using namespace ComputeMethods;


ComputeMethod::ComputeMethod(const float softening) : SOFTENING(softening) {}

//void ComputeMethod::setSystem(NbodySystem* system) {
//	this->system = system;
//}



Direct::Direct(const float softening) : ComputeMethod(softening) {}

void Direct::setSystem(NbodySystem* system) {
	this->system = system;
}

void Direct::computeAcc() {
	if(system->space == R2)
		direct2DCompute(system->device.pos_mass, system->device.acc, system->N, SOFTENING);
	else
		direct3DCompute(system->device.pos_mass, system->device.acc, system->N, SOFTENING);
}




BarnesHut::BarnesHut(int nodes, const float softening) : ComputeMethod(softening) {
	this->nodes = nodes;
}

void BarnesHut::setSystem(NbodySystem* system) {
	this->system = system;

	bodiesPerBlock = system->N; // inefficient but will do for now
	cudaMalloc(&d_bounds, sizeof(float4));
	cudaMalloc(&d_index, sizeof(int));
	cudaMalloc(&d_nodes, 4 * system->M * sizeof(int));
	cudaMalloc(&d_validBodies, 21 * bodiesPerBlock * sizeof(int));
	cudaMalloc(&d_validBodiesTop, 21 * sizeof(int));
	cudaMalloc(&d_count, system->M * sizeof(int));
	cudaMalloc(&d_pos_sorted, system->N * sizeof(float4));
	cudaMalloc(&d_idx_to_body, system->N * sizeof(int));
	cudaMalloc(&d_start, system->M * sizeof(int));
}

void BarnesHut::computeAcc() {
	barnesHutCompute(
		system->device.pos_mass, d_pos_sorted, system->device.acc, d_bounds, 
		d_index, d_nodes, d_count, d_idx_to_body, d_start, d_validBodies, d_validBodiesTop, 
		bodiesPerBlock, system->N, system->M, SOFTENING
	);
}

