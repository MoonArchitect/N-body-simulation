#include "IntegrationMethod.h"

using namespace std;
using namespace TimeSteps;
using namespace ComputeMethods;
using namespace IntegrationMethods;


// -------------------------------  Euler  -------------------------------
void Euler::integrate(vector<body>& bodies, TimeStep* timeStep, ComputeMethod* sim) {
	double dt = timeStep->getTimeStep();
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


// -------------------------------  Euler Simpletic  -------------------------------
void EulerSymplectic::integrate(vector<body>& bodies, TimeStep* timeStep, ComputeMethod* solver) {
	double dt = timeStep->getTimeStep();
	
	solver->computeAccelerations(bodies);
	
	for (body& i : bodies) {
		i.Vx += i.ax * dt;
		i.Vy += i.ay * dt;
	}
	
	for (body& i : bodies) {
		i.x += i.Vx * dt;
		i.y += i.Vy * dt;
	}
}

// -------------------------------  Verlet   -------------------------------
void Verlet::init(vector<body>& bodies, ComputeMethod* solver) {
	prevAcc = vector<vector<double>>(bodies.size());
	for (vector<double>& vec : prevAcc) {
		vec = vector<double>(2);
	}

	solver->computeAccelerations(bodies);
	for (int i = 0; i < bodies.size(); i++) {
		prevAcc[i][0] = bodies[i].ax;
		prevAcc[i][1] = bodies[i].ay;
	}
}

void Verlet::integrate(vector<body>& bodies, TimeStep* timeStep, ComputeMethod* solver) {
	double dt = timeStep->getTimeStep();
	for (int i = 0; i < bodies.size(); i++) {
		body& body = bodies[i];
		body.x += body.Vx * dt + this->prevAcc[i][0] * dt * dt / 2.0;
		body.y += body.Vy * dt + this->prevAcc[i][1] * dt * dt / 2.0;
	}
	
	solver->computeAccelerations(bodies);
	
	for (int i = 0; i < bodies.size(); i++) {
		body& body = bodies[i];
		body.Vx += (body.ax + this->prevAcc[i][0]) / 2.0 * dt;
		body.Vy += (body.ay + this->prevAcc[i][1]) / 2.0 * dt;
	}
	
	for (int i = 0; i < bodies.size(); i++) {
		this->prevAcc[i][0] = bodies[i].ax;
		this->prevAcc[i][1] = bodies[i].ay;
	}
}

// -------------------------------  ForestRuth   -------------------------------
void ForestRuth::integrate(vector<body>& bodies, TimeStep* timeStep, ComputeMethod* solver) {
	const double O = 1.35120719195966;
	double dt = timeStep->getTimeStep();

	for (body& i : bodies) {
		i.x += i.Vx * dt * O / 2;
		i.y += i.Vy * dt * O / 2;
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt * O;
		i.Vy += i.ay * dt * O;
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * (1 - O) / 2;
		i.y += i.Vy * dt * (1 - O) / 2;
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt * (1 - 2*O);
		i.Vy += i.ay * dt * (1 - 2*O);
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * (1 - O) / 2;
		i.y += i.Vy * dt * (1 - O) / 2;
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt * O;
		i.Vy += i.ay * dt * O;
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * O / 2;
		i.y += i.Vy * dt * O / 2;
	}
}

// -------------------------------  PEFRL   -------------------------------
void PEFRL::integrate(vector<body>& bodies, TimeStep* timeStep, ComputeMethod* solver) {
	const double E = 0.1786178958448091;
	const double L = -0.2123418310626054;
	const double X = -0.6626458266981849E-01;
	
	double dt = timeStep->getTimeStep();


	for (body& i : bodies) {
		i.x += i.Vx * dt * E;
		i.y += i.Vy * dt * E;
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt / 2 * (1 - 2*L);
		i.Vy += i.ay * dt / 2 * (1 - 2*L);
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * X;
		i.y += i.Vy * dt * X;
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt * L;
		i.Vy += i.ay * dt * L;
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * (1 - 2 * (X + E));
		i.y += i.Vy * dt * (1 - 2 * (X + E));
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt * L;
		i.Vy += i.ay * dt * L;
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * X;
		i.y += i.Vy * dt * X;
	}

	solver->computeAccelerations(bodies);
	for (body& i : bodies) {
		i.Vx += i.ax * dt / 2 * (1 - 2 * L);
		i.Vy += i.ay * dt / 2 * (1 - 2 * L);
	}

	for (body& i : bodies) {
		i.x += i.Vx * dt * E;
		i.y += i.Vy * dt * E;
	}
}


