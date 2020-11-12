#include "IntegrationMethod.h"

using namespace std;

//class IntegrationMethod {
/*  */
//public:
//	virtual void integrate(vector<body>& bodies, TimeStep& dt, ComputeMethod* sim) { //  void computeAcc(vector<body>& bodies)
//		throw "\'integrate\' method is Not implemented";
//	}
//};


//class Euler : public IntegrationMethod {
		
//public:
	void Euler::integrate(vector<body>& bodies, TimeStep& timeStep, ComputeMethod* sim) {
		float dt = timeStep.getTimeStep();
		for (body& i : bodies) {
			i.x += i.Vx * dt;
			i.y += i.Vy * dt;
		}

		sim->computeAccelerations(bodies);

		for (body& i : bodies) {
			i.Vx += i.ax * dt;
			i.Vy += i.ay * dt;
		}
	}
//};


//class EulerSympletic : public IntegrationMethod {

//};
