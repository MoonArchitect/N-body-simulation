#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "body.h"
#include "SystemInitializer.h"
#include "IntegrationMethod.h"
#include "ComputeMethod.h"
#include "TimeStep.h"


class Simulation {
	IntegrationMethods::IntegrationMethod* integrator;
	ComputeMethods::ComputeMethod* solver;
	TimeSteps::TimeStep* timeStep;

public:
	std::vector<body> bodies;
	bool inProgress = false;

	Simulation(IntegrationMethods::IntegrationMethod* iMethod, 
		ComputeMethods::ComputeMethod* cMethod, 
		TimeSteps::TimeStep* timeSchedule);

	void addSystem(SystemInitializers::SystemInitializer* initializer);

	void update();

	void start();
	void end();

	double getEnergy();

	double getTimeStep();
	void computeAccelerations();
};
