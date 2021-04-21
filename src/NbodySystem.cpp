#include "NbodySystem.h"
#include <stdio.h>
#include <chrono>

using namespace std;
// cudaErrCheck

NbodySystem::NbodySystem(int nBodies, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator) : compute(compute), integrator(integrator) {
    N = nBodies;
    M = compute->nodes == -1 ? N : compute->nodes;

    compute->setSystem(this);
    integrator->setSystem(this);

    h_pos_mass = new float4[M];
    h_vel = new float3[N];
    h_acc = new float3[N];

    cudaMalloc(&d_pos_mass, M * sizeof(float4));
    cudaMalloc(&d_vel, N * sizeof(float3));
    cudaMalloc(&d_acc, N * sizeof(float3));
}

int NbodySystem::getSize() { 
    return N; 
}

void NbodySystem::initSystem(int offset, int n, InitialConditions::InitialConditionsInterface* initializer) {
    initializer->initialize(offset, n, this);
}


void NbodySystem::updatePositions() {

}

void NbodySystem::computeAcceleration(bool sync) {
    compute->computeAcc();
    if (sync)
        cudaDeviceSynchronize();
}

void NbodySystem::simulate(int ticks, float dt) {
    long long totalTime = 0;
    for (int tick = 0; tick < ticks; tick++) {
        auto host_start = chrono::high_resolution_clock::now();
        
        integrator->integrate(dt);
        
        auto host_end = chrono::high_resolution_clock::now();
        long long host_duration = chrono::duration_cast<std::chrono::milliseconds>(host_end - host_start).count();
        totalTime += host_duration;
        printf("Tick %i/%i - %llims.  -  ~%lli sec. left\n", tick + 1, ticks, totalTime / (tick + 1), (ticks - tick) * totalTime / (tick + 1) / 1000);
    }
    printf("Total Host Time: %llims.\n", totalTime);
}