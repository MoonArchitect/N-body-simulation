#include <random>
#include <math.h>
#include "NbodySystem.h"

using namespace InitialConditions;


UniformBox::UniformBox(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange) 
	: pos(pos), vel(vel), size(size), massRange(massRange), velRange(velRange) {}

void UniformBox::initialize(int offset, int n, NbodySystem *system) {
	//float max_r = 40000.0f;
	//std::default_random_engine generator((unsigned int)(32423462));
	//std::uniform_real_distribution<float> dist_r(0.1f, max_r);
	//std::uniform_real_distribution<float> dist_u(-1, 1);
	//std::uniform_real_distribution<float> dist_lambda(0, 1);
	//std::uniform_real_distribution<float> dist_phi(0.0f, 2 * 3.14159265f);
	//for (int i = 0; i < n; i++) {
	//	system->h_pos_mass[i].w = 1000;
	//
	//	float phi = dist_phi(generator);
	//	float lambda = dist_lambda(generator);
	//	float u = dist_u(generator);
	//
	//	system->h_pos_mass[i].x = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * cosf(phi));
	//	system->h_pos_mass[i].y = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * sinf(phi));
	//	system->h_vel[i].x = (rand() / (float)RAND_MAX - 0.5) * 10;
	//	system->h_vel[i].y = (rand() / (float)RAND_MAX - 0.5) * 10;
	//}

	std::default_random_engine generator((unsigned int)(32423462));
	std::uniform_real_distribution<float> x_dist(pos.x - size.x / 2, pos.x + size.x / 2);
	std::uniform_real_distribution<float> y_dist(pos.y - size.y / 2, pos.y + size.y / 2);
	std::uniform_real_distribution<float> mass_dist(massRange.x, massRange.y);
	std::uniform_real_distribution<float> vel_dist(velRange.x, velRange.y);
	std::uniform_real_distribution<float> phi_dist(0.0f, 2 * 3.14159265f);
	
	for (int i = 0; i < n; i++) {
		float x = x_dist(generator);
		float y = y_dist(generator);
		float mass = mass_dist(generator);
		float phi = phi_dist(generator);
		float vel = vel_dist(generator);
		
		system->h_pos_mass[offset + i] = make_float4(x, y, 0, mass);
		system->h_vel[offset + i] = make_float3(vel * cosf(phi), vel * sinf(phi), 0);
	}


	cudaMemcpy(system->d_pos_mass, system->h_pos_mass, system->M * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(system->d_vel, system->h_vel, system->N * sizeof(float3), cudaMemcpyHostToDevice);
}