#pragma once

#include "cuda_runtime.h"

#include "IntegrationMethods.cuh"
#include "InitialConditions.cuh"
#include "ComputeMethods.h"

class NbodySystem {
public:
    int N, M;
    float4 *h_pos_mass, *d_pos_mass;
    float3 *h_vel, *h_acc, *d_vel, *d_acc;
    
    ComputeMethods::ComputeMethod* compute;
    IntegrationMethods::IntegrationMethod* integrator;


    NbodySystem(int nBodies, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator);

    int getSize();

    void initSystem(int offset, int n, InitialConditions::InitialConditionsInterface* initializer);
    void updatePositions();
    void computeAcceleration(bool sync = false);
    void simulate(int ticks, float dt);
};

