//#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


#include "NbodySystem.h"

#define SOFTENING 10.0f

int main()
{
    int n = 1024 * 512;
    float dt = 0.75f;
    std::cout.precision(30);

    //auto compute = new ComputeMethods::Direct(SOFTENING);
    auto compute = new ComputeMethods::BarnesHut(n * 4, SOFTENING);
    auto integrator = new IntegrationMethods::Euler();
    auto saver = new Callbacks::BinaryDataSaver(1, "", 500);
    NbodySystem system(n, Space::R2, compute, integrator);

    system.addSaver(saver);

    auto b1 = new InitialConditions::Standard(
        { 0, 0 },
        { 0, 0 },
        { 25000, 15000 },
        { 100, 100 },
        { 0, 0 }
    );
    system.initSystem(0, n, b1);


    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.host.pos_mass[iter * 743 + iter].x << "	V:  " << system.host.vel[iter * 743 + iter].x << std::endl;

    system.simulate(1500, dt);

    system.updateHostData();
    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.host.pos_mass[iter * 743 + iter].x << "	V:  " << system.host.vel[iter * 743 + iter].x << std::endl;


    return 0;
}

