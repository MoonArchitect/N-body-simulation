#include "ComputeMethod.h"

using namespace std;
using namespace ComputeMethods;

// -------------------------------  Direct  -------------------------------
void Direct::computeAccelerations(vector<body>& bodies) {
	const double G = 1.184 * pow(10, -4);// 6.67408 * powf(10, -11);
	int N = bodies.size();
	for (int i = 0; i < N; i++) {
		bodies[i].ax = 0;
		bodies[i].ay = 0;
	}

	for (int i = 0; i < N; i++) {
		body& bi = bodies[i];
		for (int j = i + 1; j < N; j++) {
			body& bj = bodies[j];
	
			double dx = bj.x - bi.x;
			double dy = bj.y - bi.y;
			double r = sqrt(dx * dx + dy * dy); // sqrt(pow(dx, 2) + pow(dy, 2)); // 
			//if (r < 1E-3)
			//	continue;
			
			double r3 = r * r * r; // pow(r, 3); // 
			double ki = G * bj.mass / r3;
			double kj = G * bi.mass / r3;
			
			bi.ax += ki * dx;
			bi.ay += ki * dy;
			bj.ax -= kj * dx;
			bj.ay -= kj * dy;
		}
	}
}


// -------------------------------  Direct Multi Thread  -------------------------------
