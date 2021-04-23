#include <random>
#include <math.h>
#include "NbodySystem.h"

using namespace InitialConditions;



Standard::Standard(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange)
	: pos(pos), vel(vel), size(size), massRange(massRange), velRange(velRange) {}

void Standard::initialize(int offset, int n, NbodySystem* system) {
	float max_r = 40000.0f;
	std::default_random_engine generator((unsigned int)(32423462));
	std::uniform_real_distribution<float> dist_u(-1, 1);
	std::uniform_real_distribution<float> dist_lambda(0, 1);
	std::uniform_real_distribution<float> dist_phi(0.0f, 2 * 3.14159265f);
	for (int i = 0; i < n; i++) {
		system->host.pos_mass[i].w = 1000;

		float phi = dist_phi(generator);
		float lambda = dist_lambda(generator);
		float u = dist_u(generator);

		system->host.pos_mass[i].x = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * cosf(phi));
		system->host.pos_mass[i].y = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * sinf(phi));
		system->host.pos_mass[i].z = 0;
		system->host.vel[i].x = (rand() / (float)RAND_MAX - 0.5) * 10;
		system->host.vel[i].y = (rand() / (float)RAND_MAX - 0.5) * 10;
		system->host.vel[i].z = 0;
	}

	system->updateDeviceData();
}

UniformBox::UniformBox(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange) 
	: pos(pos), vel(vel), size(size), massRange(massRange), velRange(velRange) {}

void UniformBox::initialize(int offset, int n, NbodySystem *system) {
	std::default_random_engine generator((unsigned int)(32423462));
	std::uniform_real_distribution<float> x_dist(pos.x - size.x / 2, pos.x + size.x / 2);
	std::uniform_real_distribution<float> y_dist(pos.y - size.y / 2, pos.y + size.y / 2);
	std::uniform_real_distribution<float> z_dist(pos.y - size.y / 2, pos.y + size.y / 2);
	std::uniform_real_distribution<float> mass_dist(massRange.x, massRange.y);
	std::uniform_real_distribution<float> vel_dist(velRange.x, velRange.y);
	std::uniform_real_distribution<float> phi_dist(0.0f, 2 * 3.14159265f);
	
	for (int i = 0; i < n; i++) {
		float x = x_dist(generator);
		float y = y_dist(generator);
		//float z = z_dist(generator);
		float mass = mass_dist(generator);
		float phi = phi_dist(generator);
		float v = vel_dist(generator);
		
		system->host.pos_mass[offset + i] = make_float4(x, y, 0, mass);
		system->host.vel[offset + i] = make_float4(v * cosf(phi) + vel.x, v * sinf(phi) + vel.y, 0, 0);
	}

	system->updateDeviceData();
}


UniformEllipsoid::UniformEllipsoid(float2 pos, float2 vel, float2 radius, float2 massRange, float2 velRange)
	: pos(pos), vel(vel), radius(radius), massRange(massRange), velRange(velRange) {}


void UniformEllipsoid::initialize(int offset, int n, NbodySystem* system) {
	std::default_random_engine generator((unsigned int)(32423462));
	std::uniform_real_distribution<float> u_dist(-1, 1);
	std::uniform_real_distribution<float> lambda_dist(0, 1);
	std::uniform_real_distribution<float> phi_dist(0.0f, 2 * 3.14159265f);
	std::uniform_real_distribution<float> mass_dist(massRange.x, massRange.y);
	std::uniform_real_distribution<float> vel_dist(velRange.x, velRange.y);
	for (int i = 0; i < n; i++) {

		float phi = phi_dist(generator);
		float lambda = lambda_dist(generator);
		float u = u_dist(generator);
		float mass = mass_dist(generator);
		float v = vel_dist(generator);


		system->host.pos_mass[i] = make_float4(
			radius.x * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * cosf(phi),
			radius.y * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * sinf(phi),
			0,
			mass
		);

		system->host.vel[i] = make_float4(
			v * cosf(phi) + vel.x,
			v * sinf(phi) + vel.y,
			0,
			0
		);
	}

	system->updateDeviceData();
}




