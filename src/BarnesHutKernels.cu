#include <algorithm>
#include "BarnesHutKernels.cuh"
#include "thrust/sort.h"
#include <thrust/execution_policy.h>

#define FULL_MASK 0xFFFFFFFF
#define BLOCK_SIZE 128

using namespace std;

void barnesHutCompute(float4* pos, float4* acc, float4* d_bounds, int* index, int* nodes, int* sortedIdx, int* SFCkeys, int n, int m, float theta, const float SOFTENING) {
	// Reset data
	reset << < 128, 256 >> > (pos, nodes, index, n, m);
	cudaDeviceSynchronize();
	auto errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Reset  -  %s\n", cudaGetErrorString(errorCode));
	

	// Boundaries
	compute_bounds_2D <<< 32, BLOCK_SIZE >>> (pos, d_bounds, n);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Compute Bounds 2D  -  %s\n", cudaGetErrorString(errorCode));
	

	// Build empty tree
	prebuild_tree <<< 1, 1 >>> (nodes, index, 4);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("prebuild_tree  -  %s\n", cudaGetErrorString(errorCode));

	
	// Generate SFC keys
	SFC << < 32, 512 >> > (pos, d_bounds, SFCkeys, sortedIdx, n);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("categorize  -  %s\n", cudaGetErrorString(errorCode));
	

	// Sort Bodies by SFC key
	thrust::sort_by_key(thrust::device, (unsigned int*)SFCkeys, (unsigned int*)SFCkeys + n, sortedIdx);


	// Build Quadtree
	build_quadtree <<< 32, 256 >>> (pos, d_bounds, sortedIdx, index, nodes, n, m);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Build Quadtree  -  %s\n", cudaGetErrorString(errorCode));
	

	// Find nodes centre of mass 
	centre_of_mass << < 32, 512 >> > (pos, n, m);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Mass Center  -  %s\n", cudaGetErrorString(errorCode));
	

	// Compute Force 
	compute_force << < n / 256 + 1, 256 >> > (sortedIdx, nodes, pos, acc, d_bounds, n, m, theta, SOFTENING);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Force  -  %s\n", cudaGetErrorString(errorCode));
}


void buildT(int* nodes, int node, int d, int& index) {
	if (d == 0)
		return;
	for (int i = 0; i < 4; i++)
	{
		nodes[node * 4 + i] = index; index++;
		buildT(nodes, index - 1, d - 1, index);
	}
}

__device__ static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}


__global__ void reset(float4* pos, int* d_nodes, int* d_index, int n, int m) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = gridDim.x * blockDim.x;

	if (i == 0)
		*d_index = n + 1;

	while (i < m) {
		if (i >= n)
			pos[i] = { 0, 0, 0, 0 };

		reinterpret_cast<int4*>(d_nodes)[i] = { -1, -1, -1, -1 };

		i += stride;
	}
}

__global__ void prebuild_tree(int* nodes, int* index, int d) {
	__shared__ int stack[64];
	__shared__ int depth[64];
	int top = 0;
	stack[0] = *index - 1;
	depth[0] = d;

	while (top >= 0) {
		int d = depth[top];
		int node = stack[top];

		for(int i = 0; i < 4; i++)
			nodes[node * 4 + i] = *index + i;
		

		if (d > 0) {
			for (int i = 0; i < 4; i++) {
				depth[top] = d - 1;
				stack[top++] = *index + i;
			}
		}

		*index += 4;
		top--;
	}
}

__global__ void compute_bounds_2D(float4* pos, float4* bounds, int n) {
	__shared__ float4 s_data[BLOCK_SIZE];
	int stride = gridDim.x * BLOCK_SIZE;
	int offset = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	float4 data = { pos[offset].x, pos[offset].y, pos[offset].x, pos[offset].y };

	offset += stride;
	while (offset < n) {
		data.x = fminf(data.x, pos[offset].x);
		data.y = fminf(data.y, pos[offset].y);
		data.z = fmaxf(data.z, pos[offset].x);
		data.w = fmaxf(data.w, pos[offset].y);
		offset += stride;
	}

	s_data[threadIdx.x] = data;
	__syncthreads();

	//// Alternative reduction 
	//int i = BLOCK_SIZE / 2;
	//while (i > 0) {
	//	if (threadIdx.x < i) {
	//		s_data[threadIdx.x].x = fminf(s_data[threadIdx.x].x, s_data[threadIdx.x + i].x);
	//		s_data[threadIdx.x].y = fminf(s_data[threadIdx.x].y, s_data[threadIdx.x + i].y);
	//		s_data[threadIdx.x].z = fmaxf(s_data[threadIdx.x].z, s_data[threadIdx.x + i].z);
	//		s_data[threadIdx.x].w = fmaxf(s_data[threadIdx.x].w, s_data[threadIdx.x + i].w);
	//	}
	//	__syncthreads();
	//	i /= 2;
	//}

	if (threadIdx.x == 0) {
		for (int i = 0; i < BLOCK_SIZE; i++) {
			data.x = fminf(data.x, s_data[i].x);
			data.y = fminf(data.y, s_data[i].y);
			data.z = fmaxf(data.z, s_data[i].z);
			data.w = fmaxf(data.w, s_data[i].w);
		}
		atomicMin(&(*bounds).x, data.x - fabsf(data.x) * 0.001f);
		atomicMin(&(*bounds).y, data.y - fabsf(data.y) * 0.001f);
		atomicMax(&(*bounds).z, data.z + fabsf(data.z) * 0.001f);
		atomicMax(&(*bounds).w, data.w + fabsf(data.w) * 0.001f);
	}
}

__global__ void SFC(float4* pos, float4* bounds, int* keys, int* value, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < n) {
		float l = bounds->x,
			r = bounds->z,
			b = bounds->y,
			t = bounds->w;
		float x = pos[i].x, y = pos[i].y;
		int key = 0;

		for (int i = 30; i >= 0; i -= 2) {
			float midX = (r + l) * 0.5f;
			float midY = (t + b) * 0.5f;
			if (x > midX) {
				l = midX;
				if (y > midY) {
					b = midY;
					key |= 0b01 << i;
				}
				else {
					t = midY;
					key |= 0b11 << i;
				}
			}
			else {
				r = midX;
				if (y > midY) {
					b = midY;
					// key |= 0b00 << i;
				}
				else {
					t = midY;
					key |= 0b10 << i;
				}
			}
		}

		keys[i] = key;
		value[i] = i;
		i += stride;
	}
}

__global__ void build_quadtree(float4* pos, float4* bounds, int* sortedIdx, int* index, int* nodes,int nBodies, int nNodes) {
	int size = nBodies / (blockDim.x * gridDim.x);
	int i, iter = (threadIdx.x + blockIdx.x * blockDim.x) * size;
	int end = min(iter + size, nBodies);

	float4 pi; 
	float minX, minY, maxX, maxY;
	bool newBody = true;
	int path, node, subtreeIdx = nBodies;
	int4 temp;

	while (iter < end) {
		if (newBody) {
			i = sortedIdx[iter];
			path = 0;
			node = nBodies;
			minX = bounds->x; minY = bounds->y; maxX = bounds->z; maxY = bounds->w;
			newBody = false;
			pi = pos[i];
		}
		else {
			node = nodes[subtreeIdx];
		}

		while (node >= nBodies) {
			path = 0;
		
			float midX = (maxX + minX) * 0.5f;
			if (pi.x >= midX) {
				minX = midX;
				path += 1;
			}
			else {
				maxX = midX;
			}
		
			float midY = (maxY + minY) * 0.5f;
			if (pi.y >= midY) {
				minY = midY;
				path += 2;
			}
			else {
				maxY = midY;
			}
		
			atomicAdd(&pos[node].x, pi.x * pi.w);
			atomicAdd(&pos[node].y, pi.y * pi.w);
			atomicAdd(&pos[node].w, pi.w);
		
			subtreeIdx = node * 4 + path;
			node = nodes[subtreeIdx];
		}


		if (node != -2) {
			if (atomicCAS(&nodes[subtreeIdx], node, -2) == node) {
				if (node == -1 || pi.x == pos[node].x && pi.y == pos[node].y) { // if bodies have the same position, new body is not inserted & its mass is not accounted on that level
					nodes[subtreeIdx] = i;
				}
				else {
					int locked = subtreeIdx;
					int newNode, root = nNodes;
					while (nodes[subtreeIdx] != -1)
					{
						newNode = atomicAdd(index, 1);

						if (newNode >= nNodes) {
							printf("----------------------- newNode idx (%i) is too large, try increasing # of nodes (knodes) | b1: %i(%f,%f) <-> b2: %i(%f,%f)\n", newNode, i, pi.x, pi.y, node, pos[node].x, pos[node].y);
							//return;
						}

						root = min(root, newNode);
						if (root < newNode)
							nodes[subtreeIdx] = newNode;


						path = 0;
						float midX = (maxX + minX) * 0.5f;
						float midY = (maxY + minY) * 0.5f;

						if (pos[node].x >= midX)
							path += 1;
						if (pos[node].y >= midY)
							path += 2;

						atomicAdd(&pos[newNode].x, pos[node].x * pos[node].w + pi.x * pi.w);
						atomicAdd(&pos[newNode].y, pos[node].y * pos[node].w + pi.y * pi.w);
						atomicAdd(&pos[newNode].w, pos[node].w + pi.w);

						subtreeIdx = newNode * 4 + path;
						nodes[subtreeIdx] = node;

						path = 0;
						if (pi.x >= midX) {
							minX = midX;
							path += 1;
						}
						else {
							maxX = midX;
						}

						if (pi.y >= midY) {
							minY = midY;
							path += 2;
						}
						else {
							maxY = midY;
						}

						subtreeIdx = newNode * 4 + path;
					}
					nodes[subtreeIdx] = i;
					__threadfence();
					nodes[locked] = root;
				}

				__threadfence();
				iter++;
				newBody = true;
			}
		}
		__syncthreads();
	}
}

__global__ void centre_of_mass(float4* pos, int n, int m) {
	int i = n + threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < m)
	{
		if (pos[i].w > 0) {
			pos[i].x /= pos[i].w;
			pos[i].y /= pos[i].w;
		}
		i += stride;
	}
}

__global__ void compute_force(int* sortedIdx, int* tree, float4* pos, float4* acc, float4* d_bounds, int n, int m, float theta, const float SOFTENING) {
	int bi = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float dim_size = fmaxf(d_bounds->z - d_bounds->x, d_bounds->w - d_bounds->y);
	const int inwarpId = threadIdx.x % 32;
	const int warpLane = threadIdx.x / 32 * 64;

	__shared__ int s_stack[512];
	__shared__ float s_size[512];
	__shared__ float2 s_pos[64];
	__shared__ int s_child[64];

	if (bi < n) {
		float x = pos[sortedIdx[bi]].x, y = pos[sortedIdx[bi]].y;
		float ax = 0, ay = 0;

		int top = -1;
		for (int i = 0; i < 4; i++) {
			if (tree[n * 4 + i] != -1) {
				top++;
				if (inwarpId == 0) {
					s_stack[warpLane + top] = tree[n * 4 + i];
					s_size[warpLane + top] = dim_size / theta;
					s_size[warpLane + top] *= s_size[warpLane + top];
				}
			}
		}
		
		while (top >= 0) {
			int node = s_stack[warpLane + top];
			float size = s_size[warpLane + top] * 0.25;


			if (inwarpId == 0)
				reinterpret_cast<int4*>(&s_child)[warpLane / 64] = reinterpret_cast<int4*>(tree)[node];
			
			if (inwarpId < 4 && s_child[warpLane / 16 + inwarpId] >= 0)
				s_pos[warpLane / 16 + inwarpId] = reinterpret_cast<float2*>(pos)[s_child[warpLane / 16 + inwarpId] * 2];
			

#pragma unroll
			for (int i = 0; i < 4; i++) {
				int child = s_child[warpLane / 16 + i];
				
				if (child != -1) {
					float dx = s_pos[warpLane / 16 + i].x - x;
					float dy = s_pos[warpLane / 16 + i].y - y;
					float r = SOFTENING + dx * dx + dy * dy;

					if (child < n || __all_sync(FULL_MASK, size <= r)) {
						float k = rsqrtf(r * r) * pos[child].w;
						ax += k * dx;
						ay += k * dy;
					}
					else {
						if (inwarpId == 0) {
							s_stack[warpLane + top] = child;
							s_size[warpLane + top] = size;
						}
						top++;
					}
				}
			}
			top--;
		}


		acc[sortedIdx[bi]].x = ax;
		acc[sortedIdx[bi]].y = ay;
	}
}

