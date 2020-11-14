#pragma once

#include <vector>
#include "body.h"
#include "TimeStep.h"
#include "ComputeMethod.h"

using namespace std;

class IntegrationMethod {
public:
	virtual void integrate(vector<body>& bodies, TimeStep& dt, ComputeMethod* sim) = 0;
};


class Euler : public IntegrationMethod {

public:
	void integrate(vector<body>& bodies, TimeStep& timeStep, ComputeMethod* sim) override;
};


class EulerSympletic : public IntegrationMethod {

};
