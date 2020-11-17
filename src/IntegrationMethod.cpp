#include "IntegrationMethod.h"

using namespace std;
using namespace IntegrationMethods;
using namespace ComputeMethods;
using namespace TimeSteps;



// -------------------------------  Euler  -------------------------------
void Euler::integrate(vector<body>& bodies, TimeStep& timeStep, ComputeMethod* sim) {
	double dt = timeStep.getTimeStep();
}


// -------------------------------  Euler Simpletic  -------------------------------
void EulerSympletic::integrate(vector<body>& bodies, TimeStep& timeStep, ComputeMethod* sim) {
	double dt = timeStep.getTimeStep();
	for (body& i : bodies) {
		i.x += i.Vx * dt;
		i.y += i.Vy * dt;
	}

	sim->computeAccelerations(bodies);

	for (body& i : bodies) {
		i.Vx += i.ax * dt;
		i.Vy += i.ay * dt;
	}
}


