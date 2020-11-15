#pragma once

#include <vector>
#include "body.h"
#include "TimeStep.h"
#include "ComputeMethod.h"

namespace IntegrationMethods {

	class IntegrationMethod {
	public:
		virtual void integrate(std::vector<body>& bodies, TimeSteps::TimeStep& dt, ComputeMethods::ComputeMethod* sim) = 0;
	};


	class Euler : public IntegrationMethod {

	public:
		void integrate(std::vector<body>& bodies, TimeSteps::TimeStep& timeStep, ComputeMethods::ComputeMethod* sim) override;
	};


	class EulerSympletic : public IntegrationMethod {

	};

}
