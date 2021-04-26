//#include "ComputeMethods.h"
#include "NbodySystem.h"
#include "DirectComputeKernels.cuh"
#include "BarnesHutKernels.cuh"

using namespace ComputeMethods;


ComputeMethod::ComputeMethod(const float softening) : SOFTENING(softening) {}

void ComputeMethod::setSystem(NbodySystem* system) {
	this->system = system;
}



Direct::Direct(const float softening) : ComputeMethod(softening) {}

void Direct::computeAcc() {
	if(system->space == R2)
		direct2DCompute(system->device.pos_mass, system->device.acc, system->N, SOFTENING);
	else
		direct3DCompute(system->device.pos_mass, system->device.acc, system->N, SOFTENING);
}




BarnesHut::BarnesHut(int nodes, const float softening) : ComputeMethod(softening) {
	this->nodes = nodes;
	cudaMalloc(&d_bounds, sizeof(float4));
}

void BarnesHut::computeAcc() {
	barnesHutCompute(system->device.pos_mass, system->device.acc, system->N, system->M, SOFTENING);
}

