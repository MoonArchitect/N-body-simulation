#include "TimeStep.h"

using namespace TimeSteps;


Constant::Constant(double dt) : dt(dt) {}

double Constant::getTimeStep() {
	return dt;
}