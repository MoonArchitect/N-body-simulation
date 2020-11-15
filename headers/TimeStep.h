#pragma once

namespace TimeSteps {
	
	class TimeStep {
	public:
		virtual double getTimeStep() = 0;
	};

	
	class Constant : public TimeStep {
		double dt;
	public:
		Constant(double dt);
		double getTimeStep() override;
	};


}