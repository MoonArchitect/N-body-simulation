#pragma once
#include "cuda_runtime.h"


void barnesHutCompute(float4* pos, float4* acc, float4* d_bounds, int* index, int* nodes, int* sortedIdx, int* SFCkeys, int n, int m, float theta, const float SOFTENING);

void buildT(int* nodes, int node, int d, int& index);


__device__ static float atomicMax(float* address, float val);

__device__ static float atomicMin(float* address, float val);

__global__ void reset(float4* pos, int* d_nodes, int* d_index, int n, int m);

__global__ void prebuild_tree(int* nodes, int* index, int d);

__global__ void compute_bounds_2D(float4* pos, float4* bounds, int n);

__global__ void SFC(float4* pos, float4* bounds, int* keys, int* value, int n);

__global__ void build_quadtree(float4* pos, float4* bounds, int* sortedIdx, int* index, int* nodes, int nBodies, int nNodes);

__global__ void centre_of_mass(float4* pos, int n, int m);

__global__ void compute_force(int* sortedIdx, int* tree, float4* pos, float4* acc, float4* d_bounds, int n, int m, float theta, const float SOFTENING);