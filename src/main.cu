//#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include "NbodySystem.h"

#define SOFTENING 0.5f

int main()
{
    int n = 1024 * 128;
    float dt = 0.05f;
    std::cout.precision(30);

    auto compute = new ComputeMethods::Direct(SOFTENING);
    auto integrator = new IntegrationMethods::Euler();
    NbodySystem system(n, compute, integrator);

    auto b1 = new InitialConditions::UniformBox(
        { 0, 0 },
        { 0, 0 },
        { 40000, 40000 },
        { 1000, 1000 },
        { -5, 5 }
    );
    system.initSystem(0, n, b1);

    cudaMemcpy(system.h_pos_mass, system.d_pos_mass, system.M * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(system.h_vel, system.d_vel, system.N * sizeof(float3), cudaMemcpyDeviceToHost);
    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.h_pos_mass[iter * 743 + iter].x << "	V:  " << system.h_vel[iter * 743 + iter].x << std::endl;

    system.simulate(100, dt);

    cudaMemcpy(system.h_pos_mass, system.d_pos_mass, system.M * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(system.h_vel, system.d_vel, system.N * sizeof(float3), cudaMemcpyDeviceToHost);
    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.h_pos_mass[iter * 743 + iter].x << "	V:  " << system.h_vel[iter * 743 + iter].x << std::endl;


    return 0;
}