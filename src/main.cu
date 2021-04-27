#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include "NbodySystem.h"

#define SOFTENING 1.0f

int main()
{
    int n = 1024 * 256;
    float dt = 0.01f;
    std::cout.precision(30);

    //auto compute = new ComputeMethods::Direct(SOFTENING);
    auto compute = new ComputeMethods::BarnesHut(0.4f, n * 3, SOFTENING);
    auto integrator = new IntegrationMethods::Euler();
    auto saver = new Callbacks::BinaryDataSaver(1, "", 500);
    NbodySystem system(n, Space::R2, compute, integrator);

    system.addSaver(saver);

    auto b1 = new InitialConditions::Standard(
        { 0, 0 },
        { 0, 0 },
        { 40000, 40000 },
        { 1000, 1000 },
        { 0, 0 }
    );
    system.initSystem(0, n, b1);


    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.host.pos_mass[iter * 743 + iter].x << "	V:  " << system.host.vel[iter * 743 + iter].x << std::endl;

    system.simulate(100, dt);

    system.updateHostData();
    std::cout << std::endl;
    for (int iter = 0; iter < 15; iter++)
        std::cout << system.host.pos_mass[iter * 743 + iter].x << "	V:  " << system.host.vel[iter * 743 + iter].x << std::endl;


    return 0;
}


/*

Total Host Time: 2865ms.

-0.050116240978240966796875     V:  -0.0583890192210674285888671875
3956.380126953125       V:  -26.650238037109375
-134.0438079833984375   V:  -1139.0811767578125
2167.296630859375       V:  308.468475341796875
1737.7008056640625      V:  347.460113525390625
-7679.71923828125       V:  -131.8657379150390625
738.667724609375        V:  999.94171142578125
-3697.8173828125        V:  -1052.5797119140625
-4992.3505859375        V:  -88.19792938232421875
1322.0675048828125      V:  685.317138671875
-2332.65869140625       V:  77.1967010498046875
558.70294189453125      V:  -1126.008544921875
-398.8934326171875      V:  1137.7633056640625
-1115.1077880859375     V:  -491.981231689453125
13541.86328125  V:  304.17138671875

*/