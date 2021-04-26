#include <algorithm>
#include "BarnesHutKernels.cuh"

#define BLOCK_SIZE 128
//#define SOFTENING 10.0f
#define FULL_MASK 0xffffffff
#define theta 0.8f


using namespace std;

void barnesHutCompute(float4* pos, float4* acc, int n, int m, const float SOFTENING) {
	int bodiesPerBlock = n;
	dim3 gridSize(7, 3);

	float4* d_bounds; cudaMalloc(&d_bounds, sizeof(float4)); 
	int* d_index; cudaMalloc(&d_index, sizeof(int));
	int* d_nodes; cudaMalloc(&d_nodes, 4 * m * sizeof(int));
	int* d_validBodies; cudaMalloc(&d_validBodies, 21 * bodiesPerBlock * sizeof(int));
	int* d_validBodiesTop; cudaMalloc(&d_validBodiesTop, 21 * sizeof(int));
	int* d_count; cudaMalloc(&d_count, m * sizeof(int));
	float4* d_pos_sorted; cudaMalloc(&d_pos_sorted, n * sizeof(float4));
	int* d_idx_to_body; cudaMalloc(&d_idx_to_body, n * sizeof(int));
	int* d_start; cudaMalloc(&d_start, m * sizeof(int));


	// Reset data
	reset << < 128, 512 >> > (pos, d_pos_sorted, d_start, d_nodes, d_index, d_count, d_validBodies, d_validBodiesTop, n, m);
	cudaDeviceSynchronize();
	auto errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Reset  -  %s\n", cudaGetErrorString(errorCode));
	


	// Boundaries
	compute_bounds_2D <<< 32, BLOCK_SIZE >>> (pos, n, d_bounds);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Compute Bounds 2D  -  %s\n", cudaGetErrorString(errorCode));
	


	// Build empty tree
	prebuild_tree <<< 1, 1 >>> (d_nodes, d_index, 4);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("prebuild_tree  -  %s\n", cudaGetErrorString(errorCode));

	

	/// Precompute 7x3 grid
	precompute << < 32, 512 >> > (pos, d_bounds, d_validBodies, d_validBodiesTop, bodiesPerBlock, n, 7, 3);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Precompute  -  %s\n", cudaGetErrorString(errorCode));
	


	// Build Quadtree
	build_quadtree << < gridSize, 512 >> > (d_validBodies, d_validBodiesTop, bodiesPerBlock, pos, d_index, d_nodes, d_count, d_bounds, n, m);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Build Quadtree  -  %s\n", cudaGetErrorString(errorCode));
	
	

	// Sort Bodies 
	sort_bodies << < 1, 512 >> > (pos, d_pos_sorted, d_idx_to_body, d_start, d_count, d_nodes, n, m, d_index);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Sort  -  %s\n", cudaGetErrorString(errorCode));
	


	// Find nodes centre of mass 
	centre_of_mass << < 32, 512 >> > (pos, n, m);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Mass Center  -  %s\n", cudaGetErrorString(errorCode));
	

	
	// Compute Force 
	compute_force << < n / 256 + 1, 256 >> > (d_idx_to_body, d_nodes, d_count, pos, d_pos_sorted, acc, d_bounds, n, m, SOFTENING);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Force  -  %s\n", cudaGetErrorString(errorCode));


	cudaFree(d_nodes);	cudaFree(d_count); cudaFree(d_index); cudaFree(d_validBodies); cudaFree(d_bounds); cudaFree(d_idx_to_body); cudaFree(d_start); cudaFree(d_pos_sorted);
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

__global__ void reset(float4* pos, float4* pos_sorted, int* d_start, int* d_nodes, int* d_index, int* d_count, int* d_validBodies, int* d_validBodiesTop, int n, int m) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = gridDim.x * blockDim.x;

	if (i == 0)
		*d_index = n + 1;
	if (i < 21)
		d_validBodiesTop[i] = 0;

	while (i < m) {
		if (i >= n)
			pos[i] = { 0, 0, 0, 0 };
		
		if (i < n) {
			for (int j = 0; j < 21; j++)
				d_validBodies[i * 21 + j] = 0xFEFEFEFE;
			
			pos_sorted[i] = { 0xC4C4C4C4, 0xC4C4C4C4, 0xC4C4C4C4, 0xC4C4C4C4 };
		}

		reinterpret_cast<int4*>(d_nodes)[i] = { -1, -1, -1, -1 };
		d_start[i] = 0xFFFFFFFF;

		d_count[i] = 0;

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



__global__ void compute_bounds_2D(float4* pos, int n, float4* bounds) {
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

__global__ void precompute(float4* pos, float4* bounds, int* validBodies, int* top, int allocBodies, int n, const int xDim, const int yDim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float minX = bounds->x;
	float minY = bounds->y;
	float sizeX = (bounds->z - minX) / xDim;
	float sizeY = (bounds->w - minY) / yDim;

	///* // Version 1
	while (i < n) {
		int xid = (pos[i].x - bounds->x) / sizeX;
		int yid = (pos[i].y - bounds->y) / sizeY;
		int ltop = atomicAdd(&top[xid + yid * xDim], 1);
		
		if ((xid + yid * xDim) * allocBodies + ltop > 21 * allocBodies || xid + yid * xDim > 21)
			printf("-->> %i  %i  %i  %i     %f  %f  %f  %f  %f  %f\n", i, xid, yid, (xid + yid * xDim) * allocBodies + ltop, pos[i].x, pos[i].y, bounds->x, bounds->y, sizeX, sizeY);

		validBodies[(xid + yid * xDim) * allocBodies + ltop] = i;
		i += stride;
	}
	//*/
	/*			// FIX - Some bodies are not included
	int block = threadIdx.x % 21;
	int bodyStride = blockDim.x / 21 + (blockDim.x - blockDim.x / 21 * 21) / (threadIdx.x + 1);

	__shared__ int emptyStack;
	__shared__ int exiting;
	__shared__  __align__(16) int _bodyStack[256 * 21];
	__shared__ int _top[21];
	__shared__ int _ltop[21];

	if (threadIdx.x == 0) {
		atomicExch(&emptyStack, 0);
		atomicExch(&exiting, 0);
	}
	if (threadIdx.x < 21)
		_top[threadIdx.x] = 0;

	int s_top = _top[block], xid, yid;
	while (i < n) {
		xid = (pos[i].x - minX) / sizeX;
		yid = (pos[i].y - minY) / sizeY;

		s_top = atomicAdd(&_top[xid + yid * xDim], 1);

		//if (i < 0 || i > n || s_top >= 256 || (xid + yid * xDim) * 256 + s_top >= 256 * 21)
		//	printf("There is an error here i%i |  s %i    x %f  y %f  w %f   calc %f  size %f    yid %i    xid %i    maxId %i\n", i, s_top, pos[i].x, pos[i].y, pos[i].w, (pos[i].y - minY) / sizeY, sizeY, yid, xid, 256 * 21);

		_bodyStack[(xid + yid * xDim) * 256 + s_top] = i;

		if (s_top >= 100)
			atomicExch(&emptyStack, 1);

		if (emptyStack == 1)
		{
ALL_THREAD_SYNC:
			__syncthreads();

			int s_top = _top[block];
			int bodyIdx = threadIdx.x / 21 * 4;
			int top_offset = (4 - _top[block] % 4) % 4;

			//if (s_top >= 250)
			//	printf("\n ----- Error (precompute) - s_top (%i) >= 256\n\n", s_top);

			if (threadIdx.x == 0) {
				atomicExch(&emptyStack, 0);
				atomicExch(&exiting, 1);
			}

			__syncthreads();

			if (threadIdx.x < 21) {
				_ltop[block] = atomicAdd(&top[block], _top[block] + top_offset);
				_top[block] = 0;
			}

			__syncthreads();

			while (bodyIdx < s_top) { // s_top is new empty location, so < s_top
				reinterpret_cast<int4*>(validBodies)[(block * allocBodies + _ltop[block] + bodyIdx) / 4] = reinterpret_cast<int4*>(_bodyStack)[(block * 256 + bodyIdx) / 4];
				bodyIdx += bodyStride * 4;
			}

			if (bodyIdx - bodyStride * 4 + 4 > s_top)
				for (int ii = 0; ii < top_offset; ii++)
					validBodies[block * allocBodies + _ltop[block] + s_top + ii] = -3000;

			atomicAnd(&exiting, i >= n);

			__syncthreads();
		}
		i += stride;
	}
	if (!exiting)
		goto ALL_THREAD_SYNC;
	//*/
}

__global__ void build_quadtree(int* validBodies, int* validTop, int allocBodies, float4* pos, int* index, int* nodes, int* count, float4* bounds, int nBodies, int nNodes) {
	int top = threadIdx.x;
	int block = blockIdx.x + blockIdx.y * gridDim.x;
	int i = validBodies[(blockIdx.x + blockIdx.y * gridDim.x) * allocBodies + top];
	//int i = threadIdx.x;
	int stride = blockDim.x;// *gridDim.x;

	float sizeX = (bounds->z - bounds->x) / gridDim.x;
	float sizeY = (bounds->w - bounds->y) / gridDim.y;
	bool newBody = true;
	float minX, minY, maxX, maxY;
	int path, node, subtreeIdx = nBodies;
	int depth = 0;
	while (top < allocBodies && top < validTop[block]) {
		if (i < 0 || i >= nBodies) {
			top += stride;
			i = validBodies[block * allocBodies + top];
		}
		else
		{
			if (newBody) {

				depth = 0;
				path = 0;
				node = nBodies;
				minX = bounds->x; minY = bounds->y; maxX = bounds->z; maxY = bounds->w;
				newBody = false;
			}
			else {
				node = nodes[subtreeIdx];
			}

			while (node >= nBodies) {
				path = 0; depth++;
				if (pos[i].x >= (maxX + minX) * 0.5f) {
					minX = (maxX + minX) * 0.5f; /// Check if optmized;
					path += 1;
				}
				else {
					maxX = (maxX + minX) * 0.5f;
				}
				if (pos[i].y >= (maxY + minY) * 0.5f) {
					minY = (maxY + minY) * 0.5f; /// Check if optmized;
					path += 2;
				}
				else {
					maxY = (maxY + minY) * 0.5f;
				}
				atomicAdd(&pos[node].x, pos[i].x * pos[i].w);
				atomicAdd(&pos[node].y, pos[i].y * pos[i].w);
				atomicAdd(&pos[node].w, pos[i].w);
				atomicAdd(&count[node], 1);

				subtreeIdx = node * 4 + path;
				node = nodes[subtreeIdx];
			}


			if (node != -2) {
				if (atomicCAS(&nodes[subtreeIdx], node, -2) == node) {
					if (node == -1 || pos[i].x == pos[node].x && pos[i].y == pos[node].y) { // if bodies have the same position, new body is not inserted & its mass is not accounted on that level
						nodes[subtreeIdx] = i;
					}
					else {
						int locked = subtreeIdx;
						int newNode, root = nNodes;
						while (nodes[subtreeIdx] != -1)
						{
							newNode = atomicAdd(index, 1);

							if (newNode >= nNodes) {
								printf("----------------------- newNode idx is too large b1:(%i) %i(%f,%f) <-> b2:%i(%f,%f)\n", newNode, i, pos[i].x, pos[i].y, node, pos[node].x, pos[node].y);
								return;
							}

							root = min(root, newNode);
							if (root < newNode)
								nodes[subtreeIdx] = newNode;


							path = 0;
							if (pos[node].x >= (maxX + minX) * 0.5f)
								path += 1;
							if (pos[node].y >= (maxY + minY) * 0.5f)
								path += 2;

							//atomicAdd(&pos[newNode].x, pos[node].x * pos[node].w);
							//atomicAdd(&pos[newNode].y, pos[node].y * pos[node].w);
							//atomicAdd(&pos[newNode].w, pos[node].w);
							//atomicAdd(&count[newNode], 1);
							atomicAdd(&pos[newNode].x, pos[node].x * pos[node].w + pos[i].x * pos[i].w);
							atomicAdd(&pos[newNode].y, pos[node].y * pos[node].w + pos[i].y * pos[i].w);
							atomicAdd(&pos[newNode].w, pos[node].w + pos[i].w);
							atomicAdd(&count[newNode], 2);

							subtreeIdx = newNode * 4 + path;
							nodes[subtreeIdx] = node;

							path = 0;
							if (pos[i].x >= (maxX + minX) * 0.5f) {
								minX = (maxX + minX) * 0.5f;
								path += 1;
							}
							else {
								maxX = (maxX + minX) * 0.5f;
							}

							if (pos[i].y >= (maxY + minY) * 0.5f) {
								minY = (maxY + minY) * 0.5f;
								path += 2;
							}
							else {
								maxY = (maxY + minY) * 0.5f;
							}

							//atomicAdd(&pos[newNode].x, pos[i].x * pos[i].w);
							//atomicAdd(&pos[newNode].y, pos[i].y * pos[i].w);
							//atomicAdd(&pos[newNode].w, pos[i].w);
							//atomicAdd(&count[newNode], 1);

							subtreeIdx = newNode * 4 + path;
						}
						nodes[subtreeIdx] = i;
						__threadfence();
						nodes[locked] = root;
					}
					__threadfence();
					top += stride;
					//i += stride;

					i = validBodies[(blockIdx.x + blockIdx.y * gridDim.x) * allocBodies + top];
					newBody = true;
				}
			}
			__syncthreads();
		}
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

__global__ void sort_bodies(float4* pos, float4* sorted_pos, int* idx_to_body, int* start, int* count, int* d_nodes, int n, int m, int* index) {
	int idx = n + threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	if (blockIdx.x == 0 && threadIdx.x == 0)
		start[idx] = 0;

	__syncthreads();
	while (idx < *index) {
		int s = start[idx];

		if (s >= 0) {
			for (int i = 0; i < 4; i++) {
				int node = d_nodes[idx * 4 + i];

				if (node >= n) {
					start[node] = s;
					s += count[node];
					//__threadfence();
				}
				else if (node >= 0) {
					//sorted_pos[s] = pos[node];
					idx_to_body[s] = node;
					s++;
				}
			}
			idx += stride;
		}
		__syncthreads();
	}
}

__global__ void compute_force(int* d_idx_to_body, int* tree, int* count, float4* pos, float4* sorted_pos, float4* acc, float4* d_bounds, int n, int m, const float SOFTENING) { //  , unsigned long long int* d_calcCnt, unsigned long long int* offset
	int bi = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float dim_size = fmaxf(d_bounds->z - d_bounds->x, d_bounds->w - d_bounds->y);
	const int inwarpId = threadIdx.x % 32;
	const int warpLane = threadIdx.x / 32 * 64; // --- >> * 32

	__shared__ int s_stack[512];
	__shared__ float s_size[512];
	__shared__ float2 s_pos[64];
	__shared__ int s_child[64];

	if (bi < n) {
		//float x = pos[bi].x, y = pos[bi].y;
		float x = pos[d_idx_to_body[bi]].x, y = pos[d_idx_to_body[bi]].y;
		//float x = sorted_pos[bi].x, y = sorted_pos[bi].y;
		float Fx = 0, Fy = 0;

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

		//__syncthreads();
		while (top >= 0) {
			int node = s_stack[warpLane + top];
			float size = s_size[warpLane + top] * 0.25;

			if (top >= 64) // --- >> 32
				printf("Stack is to small %i", top);

			//int _child[4];
			//reinterpret_cast<int4*>(&_child)[0] = reinterpret_cast<int4*>(tree)[node];
			if (inwarpId == 0)
				reinterpret_cast<int4*>(&s_child)[warpLane / 64] = reinterpret_cast<int4*>(tree)[node];
				//reinterpret_cast<int4*>(&s_child)[warpLane / 32] = reinterpret_cast<int4*>(tree)[node];
			if (inwarpId < 4)
				s_pos[warpLane / 16 + inwarpId] = reinterpret_cast<float2*>(pos)[s_child[warpLane / 16 + inwarpId] * 2];
				//s_pos[warpLane / 8 + inwarpId] = reinterpret_cast<float2*>(pos)[s_child[warpLane / 8 + inwarpId] * 2];

#pragma unroll
			for (int i = 0; i < 4; i++) {
				//int child = _child[i];
				int child = s_child[warpLane / 16 + i];
				//int child = s_child[warpLane / 8 + i];
				if (child != -1) {
					//float dx = pos[child].x - x;
					//float dy = pos[child].y - y;
					float dx = s_pos[warpLane / 16 + i].x - x;
					float dy = s_pos[warpLane / 16 + i].y - y;
					//float dx = s_pos[warpLane / 8 + i].x - x;
					//float dy = s_pos[warpLane / 8 + i].y - y;
					float r = SOFTENING + dx * dx + dy * dy;

					if (child < n || __all_sync(FULL_MASK, size <= r)) {
					//if (child < n) {
						float k = rsqrtf(r * r) * pos[child].w; // * r
						Fx += k * dx;
						Fy += k * dy;
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



		//force[bi].x = Fx; force[bi].y = Fy;
		acc[d_idx_to_body[bi]].x = Fx; 
		acc[d_idx_to_body[bi]].y = Fy;
		//acc[bi].x = Fx; acc[bi].y = Fy;
		//bi += stride;
	//	__syncthreads();
	}
}

