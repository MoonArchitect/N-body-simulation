#include "Simulation.h"

using namespace IntegrationMethods;
using namespace SystemInitializers;
using namespace ComputeMethods;
using namespace TimeSteps;

Simulation::Simulation(IntegrationMethod* iMethod, ComputeMethod* cMethod): integrator(iMethod), solver(cMethod) {}

void Simulation::addSystem(SystemInitializer& initializer) {
	int N = initializer.getN();
	for (int i = 0; i < N; i++)
		bodies.push_back(body());
	initializer.init(bodies);
}

void Simulation::update(TimeStep& dt) { // move TimeStep to constructor or send as a reference
	for (body& i : bodies) {
		i.ax = 0;
		i.ay = 0;
	}
	integrator->integrate(bodies, dt, solver);
}

double Simulation::getEnergy() {
	const double G = 6.67408 * pow(10.f, -11.f); // 1.184 * pow(10, -4);
	double E = 0, Ek = 0, Ep = 0;
	for (int i = 0; i < bodies.size(); i++) {
		body& bi = bodies[i];
		Ek += bi.mass * (bi.Vx * bi.Vx + bi.Vy * bi.Vy) / 2.0f;
	}

	for (int i = 0; i < bodies.size(); i++) {
		body& bi = bodies[i];
		for (int j = 0; j < bodies.size(); j++) {
			if (j == i)
				continue;
			body& bj = bodies[j];
			double r = sqrt(pow(bj.x - (double)bi.x, 2) + pow(bj.y - (double)bi.y, 2));
			Ep += -G * bi.mass * bj.mass / 2.0f / r;
		}
	}

	return Ek + Ep;
}