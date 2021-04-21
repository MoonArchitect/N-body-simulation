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
	directCompute(system->d_pos_mass, system->d_acc, system->N, SOFTENING);
}




BarnesHut::BarnesHut(int nodes, const float softening) : ComputeMethod(softening) {
	this->nodes = nodes;
}

void BarnesHut::computeAcc() {
	barnesHutCompute(system->d_pos_mass, system->d_acc, system->N, SOFTENING);
}

