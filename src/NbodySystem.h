#pragma once

#include "cuda_runtime.h"

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

    Callbacks::BinaryDataSaver* saver = nullptr;
    ComputeMethods::ComputeMethod* compute;
    IntegrationMethods::IntegrationMethod* integrator;


    NbodySystem(int nBodies, Space space, ComputeMethods::ComputeMethod* compute, IntegrationMethods::IntegrationMethod* integrator);

    int getSize();

    void initSystem(int offset, int n, InitialConditions::InitialConditionsInterface* initializer);
    void updatePositions();
    void updateHostData();
    void updateDeviceData();
    void computeAcceleration(bool sync = false);
    void simulate(int ticks, float dt);
    void addSaver(Callbacks::BinaryDataSaver* saver);
    
};

