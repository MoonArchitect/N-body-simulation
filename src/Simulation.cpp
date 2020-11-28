#include "Simulation.h"

using namespace IntegrationMethods;
using namespace SystemInitializers;
using namespace ComputeMethods;
using namespace TimeSteps;

Simulation::Simulation(IntegrationMethod* iMethod, ComputeMethod* cMethod, TimeSteps::TimeStep* timeSchedule): 
	integrator(iMethod), solver(cMethod), timeStep(timeSchedule) {}


void Simulation::start() {
	integrator->init(bodies, solver);
	solver->init(bodies);
	inProgress = true;
}

void Simulation::end() {
	inProgress = false;
	solver->close();
}

void Simulation::addSystem(SystemInitializer* initializer) {
	int N = initializer->getN();
	for (int i = 0; i < N; i++)
		bodies.push_back(body());
	initializer->init(bodies, bodies.size() - N, bodies.size());
}


double Simulation::getTimeStep() {
	return timeStep->getTimeStep();
}

void Simulation::computeAccelerations() {
	solver->computeAccelerations(bodies);
}

void Simulation::update() {
	if (!inProgress) {
		std::cout << "Start simulation before updating \n";
		throw; // make a better method
	}

	integrator->integrate(bodies, timeStep, solver);
}

double Simulation::getEnergy() {
	const double G = 1.184 * pow(10, -4); // 6.67408 * pow(10.f, -11.f); // 1.184 * pow(10, -4);
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
			double r = sqrt(pow(bj.x - bi.x, 2) + pow(bj.y - bi.y, 2));
			Ep += -G * bi.mass * bj.mass / 2.0f / r;
		}
	}

	return Ek + Ep;
}
