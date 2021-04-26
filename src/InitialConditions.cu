#include <random>
#include <math.h>
#include "NbodySystem.h"
#include <iostream>

using namespace InitialConditions;

///////////////////////////////////////// Debug Version /////////////////////////////////////////

Standard::Standard(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange)
	: pos(pos), vel(vel), size(size), massRange(massRange), velRange(velRange) {}

void Standard::initialize(int offset, int n, NbodySystem* system) {
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator((unsigned int)(234234234));
	//std::uniform_real_distribution<float> distribution(1300, 40000);
	std::exponential_distribution<float> distribution(6);
	std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);
	float* thetaArray = new float[n];
	for (int i = 0; i < n; i++) {
		float theta = distribution_theta(generator);
		thetaArray[i] = theta;
		float r = (distribution(generator) + 0.02) * 50000;
		float offsetX = 0;
		float offsetY = 0;
		if (offset == 0) {
			offsetX = 0;
			offsetY = 0;
		}
		if (i == 0) {
			system->host.pos_mass[i + offset].w = 1000000;
			system->host.pos_mass[i + offset].x = offsetX;
			system->host.pos_mass[i + offset].y = offsetY;
			system->host.pos_mass[i + offset].z = 0;
		}
		else {
			system->host.pos_mass[i + offset].w = 100;
			system->host.pos_mass[i + offset].x = system->host.pos_mass[offset].x + r * cos(theta);
			system->host.pos_mass[i + offset].y = system->host.pos_mass[offset].y + r * sin(theta);
			system->host.pos_mass[i + offset].z = sinf(r / 2000) * 2000;
		}
	}

	system->updateDeviceData();
	system->computeAcceleration(true);
	system->updateHostData();
	
	for (int i = 0; i < n; i++) {
		float rotation = offset == 0 ? 1 : -1; 
		float dx = system->host.pos_mass[0].x - system->host.pos_mass[i].x;
		float dy = system->host.pos_mass[0].y - system->host.pos_mass[i].y;
		float dz = system->host.pos_mass[0].z - system->host.pos_mass[i].z; 
		float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1);
		float Fx = system->host.acc[i + offset].x;
		float Fy = system->host.acc[i + offset].y;
		float Fz = system->host.acc[i + offset].z;
		float F = sqrtf(Fx * Fx + Fy * Fy + Fz * Fz);
		float v = sqrtf(dist * F);
		
		
		if (i == 0) {
			system->host.vel[i + offset].x = 0;
			system->host.vel[i + offset].y = 0;
			system->host.vel[i + offset].z = 0;
		}
		else {
			system->host.vel[i + offset].x = rotation * v * sin(thetaArray[i]);
			system->host.vel[i + offset].y = -rotation * v * cos(thetaArray[i]);
			system->host.vel[i + offset].z = 5;
		}
	}




	
	//float max_r = 40000.0f;
	//std::default_random_engine generator((unsigned int)(32423462));
	//std::uniform_real_distribution<float> dist_u(-1, 1);
	//std::uniform_real_distribution<float> dist_lambda(0, 1);
	//std::uniform_real_distribution<float> dist_phi(0.0f, 2 * 3.14159265f);
	//for (int i = 0; i < n; i++) {
	//	system->host.pos_mass[i].w = 1000;
	//
	//	float phi = dist_phi(generator);
	//	float lambda = dist_lambda(generator);
	//	float u = dist_u(generator);
	//
	//	system->host.pos_mass[i].x = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * cosf(phi));
	//	system->host.pos_mass[i].y = (max_r * powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * sinf(phi));
	//	system->host.pos_mass[i].z = 0;
	//	system->host.vel[i].x = (rand() / (float)RAND_MAX - 0.5) * 10;
	//	system->host.vel[i].y = (rand() / (float)RAND_MAX - 0.5) * 10;
	//	system->host.vel[i].z = 0;
	//}
	//
	system->updateDeviceData();
}

///////////////////////////////////////// UniformBox /////////////////////////////////////////

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

///////////////////////////////////////////// UniformEllipsoid /////////////////////////////////////////////

UniformEllipsoid::UniformEllipsoid(float2 pos, float2 vel, float2 radius, float2 massRange, float2 velRange)
	: pos(pos), vel(vel), radius(radius), massRange(massRange), velRange(velRange) {}


void UniformEllipsoid::initialize(int offset, int n, NbodySystem* system) {
	std::default_random_engine generator((unsigned int)(32423462));
	std::uniform_real_distribution<float> u_dist(-1, 1);
	std::uniform_real_distribution<float> lambda_dist(0.1, 1);
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
			pos.x + radius.x * cosf(phi) * lambda, //powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * cosf(phi),
			pos.y + radius.y * sinf(phi) * lambda, //powf(lambda, 1.0f / 3.0f) * sqrtf(1 - u * u) * sinf(phi),
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


///////////////////////////////////////// DiskModel //////////////////////////////////////////

DiskModel::DiskModel(float2 pos, float2 vel, float2 radius, float2 massRange, float2 velRange) 
	: UniformEllipsoid::UniformEllipsoid(pos, vel, radius, massRange, velRange) {}


void DiskModel::initialize(int offset, int n, NbodySystem* system) {
	
	UniformEllipsoid::initialize(offset, n, system);
	
	system->host.pos_mass[0] = { pos.x, pos.y, 0, 10000000 };
	system->updateDeviceData();

	system->computeAcceleration(true);

	system->updateHostData();

	for (int i = 0; i < n; i++) {
		float x = system->host.pos_mass[i].x;
		float y = system->host.pos_mass[i].y;
		float z = system->host.pos_mass[i].z;

		float r = sqrtf(x * x + y * y + z * z + 0.0001f);
		float a = sqrtf(system->host.acc[i].x * system->host.acc[i].x + system->host.acc[i].y * system->host.acc[i].y + system->host.acc[i].z * system->host.acc[i].z);

		float v = sqrtf(a * r);
		
		float vx = v * x / r;
		float vy = v * y / r;
		float vz = v * z / r;

		system->host.vel[i] = make_float4(
			-vy,
			vx,
			i == 0 ? 0 : 15,//vz,
			0
		);
	}

	system->updateDeviceData();
}
