#include "ComputeMethod.h"

using namespace std;
using namespace ComputeMethods;

// -------------------------------  Direct  -------------------------------
void Direct::computeAccelerations(vector<body>& bodies) {
	const double G = 6.67408 * pow(10, -11); // 1.184 * pow(10, -4);
	int N = bodies.size();
	for (int i = 0; i < N; i++) {
		body& bi = bodies[i];
		for (int j = i + 1; j < N; j++) {
			body& bj = bodies[j];

			double dx = bj.x - bi.x;
			double dy = bj.y - bi.y;
			double r = sqrtf(powf(dx, 2) + powf(dy, 2));
			double ki = G * bj.mass / powf(r, 3);
			double kj = G * bi.mass / powf(r, 3);

			bi.ax += ki * dx;
			bi.ay += ki * dy;
			bj.ax -= kj * dx;
			bj.ay -= kj * dy;
		}
	}
}

// -------------------------------  Direct Multi Thread  -------------------------------
void DirectMultiThread::computeAccelerations(vector<body>& bodies) {
	throw "Not implemented";
}
