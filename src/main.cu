#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include "NbodySystem.h"

#define SOFTENING 1.01f

int main()
{
    int n = 1024 * 32;
    float dt = 0.001f;
    std::cout.precision(30);

    //auto compute = new ComputeMethods::Direct(SOFTENING);
    auto compute = new ComputeMethods::BarnesHut(0.04f, 3, SOFTENING);
    auto integrator = new IntegrationMethods::Euler();
    auto saver = new Callbacks::BinaryDataSaver(5, "", 750);
    auto configSaver = new Callbacks::CheckpointSaver(500);

    NbodySystem system(n, Space::R2, compute, integrator);
    
    system.addCallback(saver);
    system.addCallback(configSaver);

    auto b1 = new InitialConditions::DiskModel(
        { 0, 0 },
        { 0, 0 },
        { 30000, 30000 },
        { 70000, 70000 },
        { 0, 0 }
    );
    system.initSystem(0, n, b1);


    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.host.pos_mass[iter * 743 + iter].x << "	V:  " << system.host.vel[iter * 743 + iter].x << std::endl;

    system.simulate(3000, dt);

    system.updateHostData();
    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.host.pos_mass[iter * 743 + iter].x << "	V:  " << system.host.vel[iter * 743 + iter].x << std::endl;


    return 0;
}

/*

Total Host Time: 57906ms.

13796.55078125  V:  16088.3310546875
-13504.6669921875       V:  -28445.69921875
11269.73046875  V:  -904.91156005859375
-3505.91552734375       V:  40165.90234375
933.03826904296875      V:  11579.1318359375
-14900.4619140625       V:  22981.896484375
19645.70703125  V:  2028.5531005859375
212.805328369140625     V:  33233.171875
-2839.16650390625       V:  15186.6640625
-2292.0302734375        V:  -37972.16015625
4925.251953125  V:  22277.458984375
-19178.32421875 V:  -22440.197265625
-27891.8828125  V:  8856.0234375
-10182.7158203125       V:  -10323.8662109375
16858.81640625  V:  24093.56640625
*/
