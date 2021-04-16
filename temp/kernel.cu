#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudnn.h>

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>


#define BLOCK_SIZE 128
#define GRID_SIZE 16
#define SOFTENING 1e+0f
#define FULL_MASK 0xffffffff
#define theta 0.7f

#define blockSize BLOCK_SIZE
#define stackSize 64
#define warp 32


typedef struct { float4* pos, * vel; } BodySystem;

void initBodies(BodySystem& p, int nBodies) {
	float max_r = 40000.0f;
	std::default_random_engine generator((unsigned int)(32423462));
	std::uniform_real_distribution<float> dist_r(0.1f, max_r);
	std::uniform_real_distribution<float> dist_u(-1, 1);
	std::uniform_real_distribution<float> dist_lambda(0, 1);
	std::uniform_real_distribution<float> dist_phi(0.0f, 2 * 3.14159265f);
	for (int i = 0; i < nBodies; i++) {
		p.pos[i].w = 1000;

		float phi = dist_phi(generator);
		float lambda = dist_lambda(generator);
		float u = dist_u(generator);

		p.pos[i].x = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * cosf(phi));
		p.pos[i].y = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * sinf(phi));
		//p.pos[i].z = (max_r * powf(lambda, 1.0f / 3.0f) * u);
		//p.pos[i].x = i - 158*512;
		//p.pos[i].y = i - 158*512;
		//p.pos[i].z = i - 158*512;

		//float r = sqrtf(powf(p.pos[i].x, 2) + powf(p.pos[i].y, 2) + powf(p.pos[i].z, 2));
		//float v = 0.1 * r;
		//p.vel[i].x = v * p.pos[i].x / r;
		//p.vel[i].y = v * p.pos[i].y / r;
		//p.vel[i].z = v * p.pos[i].z / r;
		p.vel[i].x = (rand() / (float)RAND_MAX - 0.5) * 10;
		p.vel[i].y = (rand() / (float)RAND_MAX - 0.5) * 10;
		//p.vel[i].z = 0;
		//p.vel[i].w = 0;
	}

}

__global__
void bodyForce_2d(float4* p, float4* v, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		float Fx = 0.0f; float Fy = 0.0f;// float Fz = 0.0f;

		for (unsigned int tile = 0; tile < gridDim.x; tile++) {
			__shared__ float4 spos[BLOCK_SIZE];
			//float4 tpos = p[tile * blockDim.x + threadIdx.x];
			spos[threadIdx.x] = p[tile * blockDim.x + threadIdx.x];//make_float4(tpos.x, tpos.y, tpos.z, tpos.w);
			
			
			__syncthreads();
			
			#pragma unroll
			for (int j = 0; j < BLOCK_SIZE; j++) {
				float dx = spos[j].x - p[i].x;
				float dy = spos[j].y - p[i].y;
				float distSqr = SOFTENING + dx * dx + dy * dy;
				float invDist = rsqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist * p[j].w;
				Fx += dx * invDist3; Fy += dy * invDist3;
			}
			__syncthreads();
		}

		v[i].z = Fx;
		v[i].w = Fy;
	}
}

__global__
void bodyForce_2d_I(float4* p, float4* v, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		v[i].x += dt * v[i].z; v[i].y += dt * v[i].w;
		p[i].x += v[i].x * dt; p[i].y += v[i].y * dt;
	}
}

void printArray(float* a, int n) {
	for (int i = 0; i < n; i++)
		std::cout << a[i] << "   ";
	std::cout << "\n\n\n\n";
}

void printArray(float4* a, int n, int offset = 0, char c = 'a') {
	if (c == 'a' || c == 'x') {
		for (int i = offset; i < n; i++)
			std::cout << a[i].x << "   ";
		std::cout << "\n\n\n\n";
	}
	if (c == 'a' || c == 'y') {
		for (int i = offset; i < n; i++)
			std::cout << a[i].y << "   ";
		std::cout << "\n\n\n\n";
	}
	if (c == 'a' || c == 'w') {
		for (int i = offset; i < n; i++)
			std::cout << a[i].w << "   ";
		std::cout << "\n\n\n\n";
	}	
}

void printArray(int* a, int n) {
	for (int i = 0; i < n; i++)
		std::cout << a[i] << "   ";
	std::cout << "\n\n\n\n";
}


double getKineticEnergy(BodySystem& p, int n) {
	double E = 0;

	for (int i = 0; i < n; i++) {
		double V2 = p.vel[i].x * p.vel[i].x + p.vel[i].y * p.vel[i].y;

		for (int j = i; j < n; j++) {
			if (j == i)
				continue;
			double dx = p.pos[j].x - p.pos[i].x;
			double dy = p.pos[j].y - p.pos[i].y;
			double r = sqrt(dx * dx + dy * dy);
			E += 2 * p.pos[j].w * p.pos[i].w / r;
		}

		E += V2 * p.pos[i].w / 2;
	}

	return E;
}

void compareTrees(float4* tempPos, float4* pos, int* tree1, int* tree2, int root1, int root2, int depth = 0) {
	if (root1 == -1 && root2 == -1)
		return;
	else if ((root1 < 1024 || root2 < 1024) && (root1 != root2 || pos[root1].x != tempPos[root2].x)) {
		std::cout << "E  " << root1 << "  " << root2 << "\n";
	}
	else {
		for (int i = 0; i < 4; i++)
			compareTrees(tempPos, pos, tree1, tree2, tree1[root1 * 4 + i], tree2[root2 * 4 + i], depth + 1);
	}
}

void printTree(int* tree, int root, int depth = 0) {
	if (root == -1) {
		//std::cout << "X\n";
	}
	else if (root < 512) {
		for (int j = 0; j < depth; j++)
			std::cout << "-";
		std::cout << "> " << root << "\n";
	}
	else {
		for (int j = 0; j < depth; j++)
			std::cout << "-";
		std::cout << " " << root << "\n";
		for (int i = 0; i < 4; i++) {
			printTree(tree, tree[root * 4 + i], depth + 1);
		}
	}
}

void printTreeMass(float4* pos, int* tree, int root, int depth = 0) {
	if (root == -1 || pos[root].w < 5000) {

	}
	else if (root < 512) {
		for (int j = 0; j < depth; j++)
			std::cout << "| ";
		std::cout << "> " << pos[root].w << "\n";
	}
	else {
		for (int j = 0; j < depth; j++)
			std::cout << "| ";
		std::cout << pos[root].w << "\n";
		for (int i = 0; i < 4; i++) {
			printTreeMass(pos, tree, tree[root * 4 + i], depth + 1);
		}
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

__global__
void compute_bounds_2D(float4* pos, int n, float4* bounds) {
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
		atomicMin(&(*bounds).x, data.x);
		atomicMin(&(*bounds).y, data.y);
		atomicMax(&(*bounds).z, data.z);
		atomicMax(&(*bounds).w, data.w);
	}
}

__global__
void precompute(float4* pos, float4* bounds, int* validBodies, int* top, int allocBodies, int n, const int xDim, const int yDim) {
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
		validBodies[(xid + yid * xDim) * allocBodies + ltop] = i;
		i += stride;
	}
	//*/
	/* // FIX - Some bodies are not included
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

__global__
void build_quadtree(int* validBodies, int* validTop, int allocBodies, float4* pos, int* index, int* nodes, int* count, float4* bounds, int nBodies, int nNodes) {
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

void buildT(int* nodes, int node, int d, int& index) {
	if (d == 0)
		return;
	for (int i = 0; i < 4; i++)
	{
		nodes[node * 4 + i] = index; index++;
		buildT(nodes, index - 1, d - 1, index);
	}
}


__global__
void precompute2(float4* pos, float4* bounds, int* vb, int* top, int allocBodies, int n, const int xDim, const int yDim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float minX = bounds->x;
	float minY = bounds->y;
	float sizeX = (bounds->z - minX) / xDim;
	float sizeY = (bounds->w - minY) / yDim;

	//__shared__ int a;
	//if (threadIdx.x == 0)
	//	a = 2;

	while (i < n) {
		int xid = (pos[i].x - bounds->x) / sizeX;
		int yid = (pos[i].y - bounds->y) / sizeY;
		int ltop = atomicAdd(&top[xid + yid * xDim], 1);
		vb[i] = xid + yid * xDim;
		i += stride;
	}
}

__global__ 
void precompute2_2(int* d_idx_to_body, float4* pos, float4* sorted, int* vb, int* count, int* sorted_top, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ int insert_start[21] ;
	if (threadIdx.x < 21) {
		int start = 0;
		for (int i = 0; i < threadIdx.x; i++)
			start += count[i];
	
		insert_start[threadIdx.x] = start;
	}
	
	__syncthreads();
	
	while (i < n) {
		int block = vb[i];
	
		int empty_idx = atomicAdd(&sorted_top[block], 1);
		d_idx_to_body[insert_start[block] + empty_idx] = i;
		//d_idx_to_body[i] = insert_start[block] + empty_idx;
		//sorted[insert_start[block] + empty_idx] = pos[i];
	
		i += stride;
	}
}

__global__
void build_quadtree_2(int* body_count, float4* pos, int* index, int* nodes, int* count, float4* bounds, int nBodies, int nNodes) {
	int block = blockIdx.x + blockIdx.y * gridDim.x;
	//	printf("%i  from  %i\n", -1, block);
	int bottom = 0;
	for (int i = 0; i < block; i++)
		bottom += body_count[i];
	
	int i = bottom + threadIdx.x;
	//int i = threadIdx.x + block * blockDim.x;
	
	//if(threadIdx.x == 0)
	//	atomicAdd(&count[block + 25], body_count[block]);
	
	int stride = blockDim.x;// *21;
	float sizeX = (bounds->z - bounds->x) / gridDim.x;
	float sizeY = (bounds->w - bounds->y) / gridDim.y;
	bool newBody = true;
	float minX, minY, maxX, maxY;
	int path, node, subtreeIdx = nBodies;
	int depth = 0;
	
	while (i < bottom + body_count[block]) {
	//while (i < nBodies) {
		if (i < 0 || i >= nBodies)
			printf("Error   -    %i  from  %i\n", i, block);

		if (newBody) {
			depth = 0;
			path = 0;
			node = nBodies;
			minX = bounds->x; minY = bounds->y; maxX = bounds->z; maxY = bounds->w;
			newBody = false;
			atomicAdd(&count[block], 1);
		}
		else {
			node = nodes[subtreeIdx];
		}

		while (node >= nBodies) {
			path = 0; depth++;
			if (pos[i].x >= (maxX + minX) * 0.5f) {
				minX = (maxX + minX) * 0.5f;
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
				if (node == -1) {
					nodes[subtreeIdx] = i;
				}
				else {
					int locked = subtreeIdx;
					int newNode, root = nNodes;
					while (nodes[subtreeIdx] != -1)
					{
						newNode = atomicAdd(index, 1);

						if (newNode >= nNodes) {
							printf("----------------------- newNode idx is too large (%i)\n", newNode);
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

						atomicAdd(&pos[newNode].x, pos[node].x * pos[node].w); // optimize
						atomicAdd(&pos[newNode].y, pos[node].y * pos[node].w);
						atomicAdd(&pos[newNode].w, pos[node].w);
						atomicAdd(&count[newNode], 1);

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

						atomicAdd(&pos[newNode].x, pos[i].x * pos[i].w);
						atomicAdd(&pos[newNode].y, pos[i].y * pos[i].w);
						atomicAdd(&pos[newNode].w, pos[i].w);
						atomicAdd(&count[newNode], 1);

						subtreeIdx = newNode * 4 + path;
					}

					nodes[subtreeIdx] = i;
					__threadfence();
					nodes[locked] = root;
				}
				__threadfence();
				i += stride;
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


/*
54.52 ms

5959.59521484375        V:  -6.534363269805908203125
-20307.546875   V:  1.28373610973358154296875
18551.74609375  V:  -11.0006961822509765625
11433.505859375 V:  10.1397991180419921875
-4930.3857421875        V:  8.0923099517822265625
10286.4150390625        V:  13.46554279327392578125
-30153.1875     V:  10.03342437744140625
8958.943359375  V:  1.6011226177215576171875
10118.8447265625        V:  -0.029323481023311614990234375
-15224.4970703125       V:  -4.603151798248291015625
6484.8251953125 V:  0.51446974277496337890625
-14148.6201171875       V:  10.0954494476318359375
-788.9908447265625      V:  2.121916294097900390625
-30167.009765625        V:  19.8470745086669921875
-5670.7373046875        V:  4.785671234130859375

*/

__global__ void compute_force(int* d_idx_to_body, int* tree, int* count, float4* pos, float4* sorted_pos, float2* force, float dim_size, int n, int m) { //  , unsigned long long int* d_calcCnt, unsigned long long int* offset
	int bi = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	const int inwarpId = threadIdx.x % 32;
	const int warpLane = threadIdx.x / 32 * 32;
	int warpTop = threadIdx.x / 32 * 32 - 1;

	__shared__ int s_stack[512];
	__shared__ float s_size[512];
	__shared__ float2 s_pos[64];
	__shared__ int s_child[64];

	if (bi < n) {
		//float x = pos[bi].x, y = pos[bi].y;
		float x = pos[d_idx_to_body[bi]].x, y = pos[d_idx_to_body[bi]].y;
		//float x = sorted_pos[bi].x, y = sorted_pos[bi].y;
		float Fx = 0, Fy = 0;

		//int top = -1;
		for (int i = 0; i < 4; i++) {
			if (tree[n * 4 + i] != -1) {
				//top++;
				warpTop++;
				if (inwarpId == 0) {
					//s_stack[warpLane + top] = tree[n * 4 + i];
					//s_size[warpLane + top] = dim_size / theta;
					//s_size[warpLane + top] *= s_size[warpLane + top];
					s_stack[warpTop] = tree[n * 4 + i];
					s_size[warpTop] = dim_size / theta;
					s_size[warpTop] *= s_size[warpTop];
				}
			}
		}

		//__syncthreads();
		//while (top >= 0) {
		while (warpTop >= warpLane) {
			//int node = s_stack[warpLane + top];
			//float size = s_size[warpLane + top] * 0.25;
			int node = s_stack[warpTop];
			float size = s_size[warpTop] * 0.25;
			
			//if (top >= 32)
			if (warpTop - warpLane >= 32)
				printf("Stack is to small %i", warpTop);
				//printf("Stack is to small %i", top);

			//int _child[4];
			//reinterpret_cast<int4*>(&_child)[0] = reinterpret_cast<int4*>(tree)[node];
			if(inwarpId == 0)
				reinterpret_cast<int4*>(&s_child)[warpLane / 32] = reinterpret_cast<int4*>(tree)[node];
			if(inwarpId < 4)
				s_pos[warpLane / 8 + inwarpId] = reinterpret_cast<float2*>(pos)[s_child[warpLane / 8 + inwarpId] * 2];

#pragma unroll
			for (int i = 0; i < 4; i++) {
				//int child = _child[i];
				int child = s_child[warpLane / 8 + i];
				if (child != -1) {
					//float dx = pos[child].x - x;
					//float dy = pos[child].y - y;
					float dx = s_pos[warpLane / 8 + i].x - x;
					float dy = s_pos[warpLane / 8 + i].y - y;
					float r = SOFTENING + dx * dx + dy * dy;
					
					if (child < n || __all_sync(FULL_MASK, size <= r)) {
					//if (child < n) {
						float k = rsqrtf(r * r * r) * pos[child].w;
						Fx += k * dx;
						Fy += k * dy;
					}
					else {
						if (inwarpId == 0) {
							//s_stack[warpLane + top] = child;
							//s_size[warpLane + top] = size;
							s_stack[warpTop] = child;
							s_size[warpTop] = size;
						}
						//top++;
						warpTop++;
					}
				}
			}
			//top--;
			warpTop--;
		}
		

		
		//force[bi].x = Fx; force[bi].y = Fy;
		force[d_idx_to_body[bi]].x = Fx; force[d_idx_to_body[bi]].y = Fy;
		//bi += stride;
	//	__syncthreads();
	}
}

__global__ void sort_bodies (float4* pos, float4* sorted_pos, int* idx_to_body, int* start, int* count, int* d_nodes, int n, int m, int* index) {
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


float t = 0;
int cnt = 0;

void treeForce2(BodySystem& p, int n, float dt) {
	int m = n * 4;
	int* tree = new int[4 * m];
	int* count = new int[m];
	std::fill(count, count + m, 0);
	std::fill(tree, tree + 4 * m, -1);
	int index = n + 1;
	float minX = p.pos[0].x, minY = p.pos[0].y, maxX = p.pos[0].x, maxY = p.pos[0].y;
	
	for (int i = n; i < m; i++) {
		p.pos[i].x = 0;
		p.pos[i].y = 0;
		p.pos[i].w = 0;
	}
	
	for (int i = 0; i < n; i++) {
		minX = fminf(minX, p.pos[i].x);
		minY = fminf(minY, p.pos[i].y);
		maxX = fmaxf(maxX, p.pos[i].x);
		maxY = fmaxf(maxY, p.pos[i].y);
	}
	
	
	buildT(tree, n, 6, index);
	

	const int bodiesPerBlock = 200000;
	int* d_idx_to_body; cudaMalloc(&d_idx_to_body, n * sizeof(int)); 
	float4 h_bounds = { minX - 0.1, minY - 0.1, maxX + 0.1, maxY + 0.1};
	float4* d_bounds; cudaMalloc(&d_bounds, sizeof(float4)); cudaMemcpy(d_bounds, &h_bounds, sizeof(float4), cudaMemcpyHostToDevice);
	float4* d_pos; cudaMalloc(&d_pos, m * sizeof(float4)); cudaMemcpy(d_pos, p.pos, m * sizeof(float4), cudaMemcpyHostToDevice);
	float4* d_pos_sorted; cudaMalloc(&d_pos_sorted, n * sizeof(float4)); cudaMemset(d_pos_sorted, 196, n * sizeof(float4));
	int* d_nodes; cudaMalloc(&d_nodes, 4 * m * sizeof(int)); cudaMemcpy(d_nodes, tree, 4 * m * sizeof(int), cudaMemcpyHostToDevice);
	int* d_count; cudaMalloc(&d_count, m * sizeof(int));  cudaMemcpy(d_count, count, m * sizeof(int), cudaMemcpyHostToDevice);
	int* d_index; cudaMalloc(&d_index, sizeof(int));  cudaMemcpy(d_index, &index, sizeof(int), cudaMemcpyHostToDevice);
	int* d_validBodies; cudaMalloc(&d_validBodies, 21 * bodiesPerBlock * sizeof(int)); cudaMemset(d_validBodies, 254, 21 * bodiesPerBlock * sizeof(int));
	int* d_validBodiesTop; cudaMalloc(&d_validBodiesTop, 21 * sizeof(int)); cudaMemset(d_validBodiesTop, 0, 21 * sizeof(int));
	int* d_validBodies2; cudaMalloc(&d_validBodies2, 21 * bodiesPerBlock * sizeof(int)); cudaMemset(d_validBodies2, 254, 21 * bodiesPerBlock * sizeof(int));
	int* d_validBodiesTop2; cudaMalloc(&d_validBodiesTop2, 21 * sizeof(int)); cudaMemset(d_validBodiesTop2, 0, 21 * sizeof(int));
	int* d_sorted_top; cudaMalloc(&d_sorted_top, 21 * sizeof(int)); cudaMemset(d_sorted_top, 0, 21 * sizeof(int));
	int* d_start; cudaMalloc(&d_start, m * sizeof(int)); cudaMemset(d_start, 255, m * sizeof(int));
	auto errorCode = cudaGetLastError();
	dim3 gridSize(7, 3);
	float2* d_force, *force = new float2[n]; cudaMalloc(&d_force, sizeof(float2) * n);
	
	
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	precompute <<< 32, 512  >>> (d_pos, d_bounds, d_validBodies, d_validBodiesTop, bodiesPerBlock, n, 7, 3);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Precompute  -  %s\n", cudaGetErrorString(errorCode));

	
	//precompute2 <<< 32, 512  >>> (d_pos, d_bounds, d_validBodies2, d_validBodiesTop2, bodiesPerBlock, n, 7, 3);
	//cudaDeviceSynchronize();
	//errorCode = cudaGetLastError();
	//if (cudaSuccess != errorCode)
	//	printf("2  -  %s\n", cudaGetErrorString(errorCode));
	//
	//
	//precompute2_2 <<< 32, 512  >>> (d_idx_to_body, d_pos, d_pos_sorted, d_validBodies2, d_validBodiesTop2, d_sorted_top, n);
	//cudaDeviceSynchronize();
	//errorCode = cudaGetLastError();
	//if (cudaSuccess != errorCode)
	//	printf("2.2  -  %s\n", cudaGetErrorString(errorCode));


	build_quadtree <<< gridSize, 512 >>> (d_validBodies, d_validBodiesTop, bodiesPerBlock, d_pos, d_index, d_nodes, d_count, d_bounds, n, m); 
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Build Quadtree  -  %s\n", cudaGetErrorString(errorCode));
	
	sort_bodies << < 1, 512 >> > (d_pos, d_pos_sorted, d_idx_to_body, d_start, d_count, d_nodes, n, m, d_index);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Sort  -  %s\n", cudaGetErrorString(errorCode));


	centre_of_mass << < 32, 512 >> > (d_pos, n, m);
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Mass Center  -  %s\n", cudaGetErrorString(errorCode));
	
	float size = fmaxf(maxX - minX, maxY - minY);
	//unsigned long long int* d_calcCnt; cudaMalloc(&d_calcCnt, sizeof(unsigned long long int)); cudaMemset(d_calcCnt, 0, sizeof(unsigned long long int));
	//unsigned long long int* d_offset; cudaMalloc(&d_offset, sizeof(unsigned long long int)); cudaMemset(d_offset, 0, sizeof(unsigned long long int));
	compute_force << < n / 256, 256>> > (d_idx_to_body, d_nodes, d_count, d_pos, d_pos_sorted, d_force, size, n, m); // , d_calcCnt, d_offset
	cudaDeviceSynchronize();
	errorCode = cudaGetLastError();
	if (cudaSuccess != errorCode)
		printf("Force  -  %s\n", cudaGetErrorString(errorCode));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	t += elapsedTime;
	cnt++;
	//unsigned long long int calcCnt; cudaMemcpy(&calcCnt, d_calcCnt, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	//unsigned long long int offset; cudaMemcpy(&offset, d_offset, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	if (cnt == 50)
		std::cout << "\t\t------> " << t / cnt << " ms\n";
		//std::cout << t / cnt << "ms  " << calcCnt - offset << "\n";
	//std::cout << calcCnt << "\n";


	cudaMemcpy(force, d_force, n * sizeof(float2), cudaMemcpyDeviceToHost);
	//cudaMemcpy(tree, d_nodes, 4 * m * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(count, d_count, m  * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(p.pos, d_pos, m  * sizeof(float4), cudaMemcpyDeviceToHost);
	//cudaMemcpy(p.pos, d_pos_sorted, m  * sizeof(float4), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
	
	
	for (int i = 0; i < n; i++) { 
		p.vel[i].x += force[i].x * dt;
		p.vel[i].y += force[i].y * dt;
		//p.vel[i].x += force[idx_to_body[i]].x * dt;
		//p.vel[i].y += force[idx_to_body[i]].y * dt;
	}

	/*
	// Build quadtree
	float inX = minX, inY = minY, axX = maxX, axY = maxY;
	for (int i = 0; i < n; i++) {
		float minX = inX, minY = inY, maxX = axX, maxY = axY;
		int path = 0, node = n, parentIdx = 0;

		do {
			path = 0;
			if (p.pos[i].x >= (maxX + minX) * 0.5) {
				minX = (maxX + minX) * 0.5; /// Check if optmized;
				path += 1;
			}
			else {
				maxX = (maxX + minX) * 0.5;
			}
			if (p.pos[i].y >= (maxY + minY) * 0.5) {
				minY = (maxY + minY) * 0.5; /// Check if optmized;
				path += 2;
			}
			else {
				maxY = (maxY + minY) * 0.5;
			}

			p.pos[node].x += p.pos[i].x * p.pos[i].w;
			p.pos[node].y += p.pos[i].y * p.pos[i].w;
			p.pos[node].w += p.pos[i].w;
			count[node] += 1;

			parentIdx = node * 4 + path;
			node = tree[parentIdx];
		} while (node >= n);

		int newNode = 0;
		while (tree[parentIdx] != -1) {
			newNode = index; index++;
			tree[parentIdx] = newNode;

			path = 0;
			if (p.pos[node].x >= (maxX + minX) * 0.5)
				path += 1;
			if (p.pos[node].y >= (maxY + minY) * 0.5)
				path += 2;

			p.pos[newNode].x += p.pos[node].x * p.pos[node].w;
			p.pos[newNode].y += p.pos[node].y * p.pos[node].w;
			p.pos[newNode].w += p.pos[node].w;
			count[newNode] += 1;

			parentIdx = newNode * 4 + path;
			tree[parentIdx] = node;

			path = 0;
			if (p.pos[i].x >= (maxX + minX) * 0.5) {
				minX = (maxX + minX) * 0.5;
				path += 1;
			}
			else {
				maxX = (maxX + minX) * 0.5;
			}

			if (p.pos[i].y >= (maxY + minY) * 0.5) {
				minY = (maxY + minY) * 0.5;
				path += 2;
			}
			else {
				maxY = (maxY + minY) * 0.5;
			}

			p.pos[newNode].x += p.pos[i].x * p.pos[i].w;
			p.pos[newNode].y += p.pos[i].y * p.pos[i].w;
			p.pos[newNode].w += p.pos[i].w;
			count[newNode] += 1;

			parentIdx = newNode * 4 + path;
		}

		tree[parentIdx] = i;
	}

	//*/
	
	// Compute centre of mass 
	//for (int i = n; i < index; i++) {
	//	if (p.pos[i].w != 0) {
	//		p.pos[i].x /= p.pos[i].w;
	//		p.pos[i].y /= p.pos[i].w;
	//	}
	//}

	/*
	float F[2] = { 0, 0 };
	int* stack = new int[128];
	float* depth = new float[128];
	for (int bi = 0; bi < n; bi++) {
		F[0] = 0; F[1] = 0;
	
		float size = fmaxf(maxX - minX, maxY - minY);
		int top = -1;
	
		for (int i = 0; i < 4; i++) {
			if (tree[n * 4 + i] != -1) {
				stack[i] = tree[n * 4 + i];
				//if (theta == 0)
					depth[i] = 99999999;
				//else
				//	depth[i] = size * size / (theta * theta);
				top++;
			}
		}
		while (top >= 0) {
			int node = stack[top];
			float d = depth[top] * 0.25f;
			for (int i = 0; i < 4; i++) {
				int ch = tree[node * 4 + i];
				if (ch != -1) {
					double dx = p.pos[ch].x - p.pos[bi].x;
					double dy = p.pos[ch].y - p.pos[bi].y;
					double r = SOFTENING + dx * dx + dy * dy;
					if (ch < n || d <= r) {
						r = rsqrt(r);
						double k = p.pos[ch].w * r * r * r;
						F[0] += dx * k;
						F[1] += dy * k;
					}
					else {
						stack[top] = ch;
						depth[top] = d;
						top++;
					}
				}
			}
			top--;
		}
		
		p.vel[bi].x += F[0] * dt; p.vel[bi].y += F[1] * dt;
		//p.vel[idx_to_body[bi]].x += F[0] * dt; p.vel[idx_to_body[bi]].y += F[1] * dt;
	}
	//*/
	for (int bi = 0; bi < n; bi++) {
		p.pos[bi].x += p.vel[bi].x * dt; 
		p.pos[bi].y += p.vel[bi].y * dt;
		//p.pos[bi].x = pos[bi].x + p.vel[bi].x * dt; 
		//p.pos[bi].y = pos[bi].y + p.vel[bi].y * dt;
	}
	
	//free(pos);
	cudaFree(d_pos); cudaFree(d_nodes);	cudaFree(d_count); cudaFree(d_index); cudaFree(d_bounds); cudaFree(d_validBodies); cudaFree(d_force); cudaFree(d_validBodies2);  cudaFree(d_idx_to_body); cudaFree(d_start);
	cudaFree(d_pos_sorted);

	free(force);
	free(tree);
	free(count);
}


int main(const int argc, const char** argv) {
	float dt = 0.05f; 
	const int nBodies = 1024 * 1024,
		nIters = 100;
	
	#pragma region Init

	std::cout.precision(30);
	const int nNodes = nBodies * 5, 
		bytes = 2 * nNodes * sizeof(float4);
	
	float* buf = (float*)malloc(bytes);
	std::fill(buf, buf + 8 * nNodes, 0.0f);;
	BodySystem p = { (float4*)buf, ((float4*)buf) + nNodes };
	
	float* d_buf; cudaMalloc(&d_buf, bytes);
	BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nNodes };
	
	initBodies(p, nBodies);
	cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
	
	std::cout << std::endl;
	for (int iter = 0; iter < 15; iter++)
		std::cout << p.pos[iter * 743 + iter].x << "	V:  " << p.vel[iter * 743 + iter].x << std::endl;

	int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
	double totalTime = 0.0;// , initEnergy = getKineticEnergy(p, nBodies);

	#pragma endregion

	for (int iter = 1; iter <= nIters; iter++) {
		auto t1 = std::chrono::high_resolution_clock::now();

		//bodyForce_2d <<< nBlocks, BLOCK_SIZE >>> (d_p.pos, d_p.vel, dt, nBodies);
		//bodyForce_2d_I <<< nBlocks, BLOCK_SIZE >>> (d_p.pos, d_p.vel, dt, nBodies);
		//cudaDeviceSynchronize();
		//cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
		
		treeForce2(p, nBodies, dt);

		/*

5963.0625       V:  -5.945094585418701171875
-20282.787109375        V:  1.20915710926055908203125
18560.28125     V:  0.284204781055450439453125
11434.0859375   V:  7.091912746429443359375
-4926.599609375 V:  14.097530364990234375
10256.9033203125        V:  -7.693981647491455078125
-30163.216796875        V:  6.04022216796875
8951.0400390625 V:  -1.4506309032440185546875
10111.92578125  V:  4.120280742645263671875
-15222.146484375        V:  -3.2207539081573486328125
6488.6025390625 V:  -1.63900363445281982421875
-14119.2685546875       V:  14.0123462677001953125
-783.9073486328125      V:  3.0803344249725341796875
-30183.09375    V:  3.6621448993682861328125
-5672.23828125  V:  -8.7071399688720703125
		
		*/
		
		//cudaEvent_t startEvent, stop; float elapsedTime;
		//cudaEventCreate(&startEvent);
		//cudaEventCreate(&stop);
		//cudaEventRecord(startEvent, 0);
		
		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&elapsedTime, startEvent, stop);
		//cudaEventDestroy(startEvent);
		//cudaEventDestroy(stop);
		//std::cout << elapsedTime << std::endl << std::endl;
		
		//cudaDeviceSynchronize();
		if (iter % 1 == 0) {
			//std::cout << abs(initEnergy - getKineticEnergy(p, nBodies)) / initEnergy << "\n";
			auto t2 = std::chrono::high_resolution_clock::now();
			auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0; totalTime += dt;
			printf("Iteration %d: %.3f seconds, Left: %d sec\n", iter, dt, (int)((nIters - iter) / (float)iter * totalTime)); t1 = std::chrono::high_resolution_clock::now();
		}
	}
	double avgTime = totalTime / (double)(nIters - 1);



	printf("Average rate for iterations 2 through %d: %.3f +- %.3f steps per second.\n", nIters);
	printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

	std::cout << std::endl;
	for (int iter = 0; iter < 15; iter++)
		std::cout << p.pos[iter * 743 + iter].x << "	V:  " << p.vel[iter * 743 + iter].x << std::endl;
	std::cout << std::endl;

	free(buf);
	cudaFree(d_buf);
}
