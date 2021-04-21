#pragma once
#include "cuda_runtime.h"


void barnesHutCompute(float4* pos, float3* acc, int n, const float SOFTENING);