#include "Simulation.h"

//class Simulation {
	/*  */
//	IntegrationMethod* integrator;
//	ComputeMethod* solver;
//	std::vector<body> bodies;

//public:
	//Simulation(IntegrationMethod::IntegrationMethod* iMethod, ComputeMethod::ComputeMethod* cMethod) : integrator(iMethod), solver(cMethod) {}

Simulation::Simulation(IntegrationMethod* iMethod, ComputeMethod* cMethod): integrator(iMethod), solver(cMethod) {}


void Simulation::addSystem(SystemInitializer& initializer) {
	int N = initializer.getN();
	for (int i = 0; i < N; i++)
		this->bodies.push_back(body());
	initializer.init(this->bodies);
}

void Simulation::update(TimeStep& dt) { // move TimeStep to constructor or send as a reference
	for (body& i : this->bodies) {
		i.ax = 0;
		i.ay = 0;
	}
	this->integrator->integrate(this->bodies, dt, this->solver);
}
//};