#pragma once

#include <vector>
#include "body.h"
#include "SystemInitializer.h"
#include "IntegrationMethod.h"
#include "ComputeMethod.h"
#include "TimeStep.h"



class Simulation {
	/*  */
	IntegrationMethods::IntegrationMethod* integrator;
	ComputeMethods::ComputeMethod* solver;

public:
	std::vector<body> bodies;

	Simulation(IntegrationMethods::IntegrationMethod* a, ComputeMethods::ComputeMethod* b);

	void addSystem(SystemInitializers::SystemInitializer& initializer);

	void update(TimeSteps::TimeStep& dt);

	double getEnergy();
};