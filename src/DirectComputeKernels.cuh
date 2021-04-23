#pragma once
#include "cuda_runtime.h"


__global__ void direct2DComputeKernel(float4* pos, float4* acc, int n, const float SOFTENING);

void direct2DCompute(float4* pos, float4* acc, int n, const float SOFTENING);


__global__ void direct3DComputeKernel(float4* pos, float4* acc, int n, const float SOFTENING);

void direct3DCompute(float4* pos, float4* acc, int n, const float SOFTENING);
