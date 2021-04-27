#pragma once
#include "cuda_runtime.h"


void barnesHutCompute(
	float4* pos, float4* acc, float4* d_bounds,
	int* d_index, int* d_nodes, int* d_count, int* d_idx_to_body, int* d_start, int* d_validBodies, int* d_validBodiesTop,
	int bodiesPerBlock, int n, int m, float theta, const float SOFTENING
);

void buildT(int* nodes, int node, int d, int& index);

__device__ static float atomicMax(float* address, float val);

__device__ static float atomicMin(float* address, float val);

__global__ void reset(float4* pos, int* d_start, int* d_nodes, int* d_index, int* d_count, int* d_validBodies, int* d_validBodiesTop, int n, int m);

__global__ void prebuild_tree(int* nodes, int* index, int d);

__global__ void compute_bounds_2D(float4* pos, int n, float4* bounds);

__global__ void precompute(float4* pos, float4* bounds, int* validBodies, int* top, int allocBodies, int n, const int xDim, const int yDim);

__global__ void build_quadtree(int* validBodies, int* validTop, int allocBodies, float4* pos, int* index, int* nodes, int* count, float4* bounds, int nBodies, int nNodes);

__global__ void centre_of_mass(float4* pos, int n, int m);

__global__ void sort_bodies(float4* pos, int* idx_to_body, int* start, int* count, int* d_nodes, int n, int m, int* index);

__global__ void compute_force(int* d_idx_to_body, int* tree, int* count, float4* pos, float4* acc, float4* d_bounds, int n, int m, float theta, const float SOFTENING);