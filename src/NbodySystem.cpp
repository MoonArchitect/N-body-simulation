#include "NbodySystem.h"
#include <stdio.h>
#include <chrono>

using namespace std;
// cudaErrCheck

NbodySystem::NbodySystem(int nBodies, Space space, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator) : compute(compute), integrator(integrator) {
    N = nBodies;
    M = compute->nodes == -1 ? N : compute->nodes;
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

int NbodySystem::getSize() { 
    return N; 
}

void NbodySystem::initSystem(int offset, int n, InitialConditions::InitialConditionsInterface* initializer) {
    initializer->initialize(offset, n, this);
}

void NbodySystem::updatePositions() {

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

void NbodySystem::addSaver(Callbacks::BinaryDataSaver* saver) {
    this->saver = saver;
}

void NbodySystem::simulate(int ticks, float dt) {
    long long totalTime = 0;
    if (saver != nullptr)
        saver->reset(this);

    for (int tick = 0; tick < ticks; tick++) {
        auto host_start = chrono::high_resolution_clock::now();
        
        integrator->integrate(dt);
        
        auto host_end = chrono::high_resolution_clock::now();
        long long host_duration = chrono::duration_cast<std::chrono::milliseconds>(host_end - host_start).count();
        totalTime += host_duration;

        
        if (tick % 2 == 0)
            printf("Tick %i/%i - %llims.  -  ~%lli sec. left\n", tick + 1, ticks, totalTime / (tick + 1), (ticks - tick) * totalTime / (tick + 1) / 1000);
        if (saver != nullptr)
            saver->end(tick, this);
    }
    printf("Total Host Time: %llims.\n", totalTime);
}