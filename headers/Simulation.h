#pragma once

#include <vector>
#include "body.h"
#include "SystemInitializer.h"
#include "IntegrationMethod.h"
#include "ComputeMethod.h"
#include "TimeStep.h"

using namespace std;

class Simulation {
	/*  */
	IntegrationMethod* integrator;
	ComputeMethod* solver;

public:
	vector<body> bodies;

	Simulation(IntegrationMethod* a, ComputeMethod* b);

	void addSystem(SystemInitializer& initializer);

	void update(TimeStep& dt);
};