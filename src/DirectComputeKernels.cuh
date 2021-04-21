#pragma once
#include "cuda_runtime.h"


__global__ void directComputeKernel(float4* pos, float3* acc, int n, const float SOFTENING);

void directCompute(float4* pos, float3* acc, int n, const float SOFTENING);