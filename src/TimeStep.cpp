#include "TimeStep.h"

//class TimeStep {
//public:
//	virtual float getTimeStep() {
//		throw "Not Impleneted";
//	}
//};

//class Constant : TimeStep {
//	float dt;
//public:
Constant::Constant(float dt) : dt(dt) {}

float Constant::getTimeStep() {
	return this->dt;
}
//};