#include <stdio.h>
#include <chrono>
#include <filesystem>

#include "NbodySystem.h"

using namespace std;
// cudaErrCheck

NbodySystem::NbodySystem(int nBodies, Space space, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator) : compute(compute), integrator(integrator) {
	N = nBodies;
	M = compute->knodes * N;
	this->space = space;

	compute->setSystem(this);
	integrator->setSystem(this);

	host.pos_mass = new float4[M];
	host.vel = new float4[N];
	host.acc = new float4[N];

	cudaMalloc(&device.pos_mass, M * sizeof(float4));
	cudaMalloc(&device.vel, N * sizeof(float4));
	cudaMalloc(&device.acc, N * sizeof(float4));
}

NbodySystem::NbodySystem(string configPath, Space space, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator)
	: compute(compute), integrator(integrator) {

	if (!filesystem::is_regular_file(configPath) || filesystem::path(configPath).extension() == ".param") {
		printf("Config read failed. Make sure \'configPath\' is a .param file\n");
		throw "Config read failed. Make sure \'configPath\' is a .param file";
	}

	//auto json = fspath.parent_path().string() + fspath.filename().string() + ".json";
	//if (!filesystem::exists(json)) {
	//    printf("JSON Configuration file is not found. Make sure .json is in the same directory as .param file\n");
	//    throw "JSON Configuration file is not found. Make sure .json is in the same directory as .param file";
	//}
	
	N = filesystem::file_size(configPath) / 28;
	this->space = space;
	M = compute->knodes * N;

	compute->setSystem(this);
	integrator->setSystem(this);

	host.pos_mass = new float4[M];
	host.vel = new float4[N];
	host.acc = new float4[N];

	cudaMalloc(&device.pos_mass, M * sizeof(float4));
	cudaMalloc(&device.vel, N * sizeof(float4));
	cudaMalloc(&device.acc, N * sizeof(float4));

	ifstream dataFile(configPath, ios_base::binary);
	
	const int bufferSize = 10000; float buffer[bufferSize * 7];
	for (int i = 0; i < N; i += bufferSize) {
		dataFile.read((char*)&buffer, bufferSize * 28);
		for (int j = 0; j < dataFile.gcount() / 28; j++) {
			host.pos_mass[i + j] = { buffer[j * 7], buffer[j * 7 + 1], buffer[j * 7 + 2], buffer[j * 7 + 3] };
			host.vel[i + j] = { buffer[j * 7 + 4], buffer[j * 7 + 5], buffer[j * 7 + 6], 0 };
		}
	}
	
	dataFile.close();

	updateDeviceData();
}


void NbodySystem::initSystem(int offset, int n, InitialConditions::InitialConditions* initializer) {
	initializer->initialize(offset, n, this);
}

void NbodySystem::updateHostData() {
	cudaMemcpy(host.pos_mass, device.pos_mass, M * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(host.vel, device.vel, N * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(host.acc, device.acc, N * sizeof(float4), cudaMemcpyDeviceToHost);
}

void NbodySystem::updateDeviceData() {
	cudaMemcpy(device.pos_mass, host.pos_mass, M * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(device.vel, host.vel, N * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(device.acc, host.acc, N * sizeof(float4), cudaMemcpyHostToDevice);
}

void NbodySystem::computeAcceleration(bool sync) {
	compute->computeAcc();
	if (sync)
		cudaDeviceSynchronize();
}

void NbodySystem::addCallback(Callbacks::Callback* callback) {
	callbacks.push_back(callback);
}

void NbodySystem::simulate(int ticks, float dt) {
	long long totalTime = 0;
	for (Callbacks::Callback* callback : this->callbacks)
		callback->reset(this);

	chrono::steady_clock::time_point host_start, host_end;
	host_start = chrono::high_resolution_clock::now();

	for (int tick = 0; tick < ticks; tick++) {
		for (Callbacks::Callback* callback : this->callbacks)
			callback->start(tick, this);



		integrator->integrate(dt);


		host_end = chrono::high_resolution_clock::now();
		long long host_duration = chrono::duration_cast<std::chrono::milliseconds>(host_end - host_start).count();
		if (host_duration > 250) {
			totalTime += host_duration;
			host_start = chrono::high_resolution_clock::now();
			string str; str.resize(20, '-'); fill_n(str.begin(), (tick) * 20 / ticks + 1, '#');
			printf("Tick %i/%i - [%s] %i%% - %llims/tick  -  ~%lli sec. left\n",
				tick + 1, ticks, str.c_str(), (tick + 1) * 100 / ticks, totalTime / (tick + 1), (ticks - tick) * totalTime / (tick + 1) / 1000);
		}
		for (Callbacks::Callback* callback : this->callbacks)
			callback->end(tick, this);
	}
	printf("Total Host Time: %llims.\n", totalTime);
}