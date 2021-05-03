#pragma once

#include "cuda_runtime.h"

class NbodySystem; 
enum Space;

namespace SimulationStats {

	std::tuple<double, double, double> computeLinearMomentum(Space space, float4* pos, float4* vel, int n);

	__global__ void compute_LMoment_kE_R2Kernel(double* momentum, double* kE, float4* pos, float4* vel, int n);

	__global__ void compute_LMoment_kE_R3Kernel(double* momentum, double* kE, float4* pos, float4* vel, int n);

	__global__ void compute_pE_Kernel(double* pE, float4* pos, int n);
}
