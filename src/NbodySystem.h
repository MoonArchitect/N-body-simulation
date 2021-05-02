#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "IntegrationMethods.cuh"
#include "InitialConditions.cuh"
#include "ComputeMethods.h"
#include "Callbacks.h"

enum Space {
    R2,
    R3
};

static struct dataBuffer {
    float4* pos_mass;
    float4* vel, *acc;
};

class NbodySystem {
public:
    int N, M;
    Space space;
    dataBuffer host, device;
    

    std::vector<Callbacks::Callback*> callbacks;
    ComputeMethods::ComputeMethod* compute;
    IntegrationMethods::IntegrationMethod* integrator;

    //std::filesystem::space_info a;

    NbodySystem(int nBodies, Space space, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator);
    NbodySystem(std::string configPath, Space space, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator);

    void initSystem(int offset, int n, InitialConditions::InitialConditions* initializer);
    void updateHostData();
    void updateDeviceData();
    void computeAcceleration(bool sync = false);
    void simulate(int ticks, float dt);
    void addCallback(Callbacks::Callback* callback);
    
};

