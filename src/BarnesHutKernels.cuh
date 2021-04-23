#pragma once
#include "cuda_runtime.h"


void barnesHutCompute(float4* pos, float4* acc, int n, const float SOFTENING);