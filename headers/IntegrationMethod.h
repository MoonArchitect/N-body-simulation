#pragma once

#include <vector>
#include "body.h"
#include "TimeStep.h"
#include "ComputeMethod.h"

namespace IntegrationMethods {
	class IntegrationMethod {
	public:
		virtual void init(std::vector<body>& bodies, ComputeMethods::ComputeMethod* solver) {};
		virtual void integrate(std::vector<body>& bodies, TimeSteps::TimeStep* timeStep, ComputeMethods::ComputeMethod* solver) = 0;
	};


	class Euler : public IntegrationMethod {
	public:
		void integrate(std::vector<body>& bodies, TimeSteps::TimeStep* timeStep, ComputeMethods::ComputeMethod* solver) override;
	};


	class EulerSymplectic : public IntegrationMethod {
	public:
		void integrate(std::vector<body>& bodies, TimeSteps::TimeStep* timeStep, ComputeMethods::ComputeMethod* solver) override;
	};

	class Verlet : public IntegrationMethod {
		std::vector<std::vector<double>> prevAcc;
	
	public:
		void init(std::vector<body>& bodies, ComputeMethods::ComputeMethod* solver) override;
		void integrate(std::vector<body>& bodies, TimeSteps::TimeStep* timeStep, ComputeMethods::ComputeMethod* solver) override;
	};

	class ForestRuth : public IntegrationMethod {
	public:
		void integrate(std::vector<body>& bodies, TimeSteps::TimeStep* timeStep, ComputeMethods::ComputeMethod* solver) override;
	};

	class PEFRL : public IntegrationMethod {
	public:
		void integrate(std::vector<body>& bodies, TimeSteps::TimeStep* timeStep, ComputeMethods::ComputeMethod* solver) override;
	};
}
