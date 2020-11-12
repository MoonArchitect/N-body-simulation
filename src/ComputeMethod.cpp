#include "ComputeMethod.h"

using namespace std;
	
//class ComputeMethod {
//public:
//	virtual void computeAccelerations(vector<body>& bodies) {
//		throw "\'computeAccelerations\' method is Not implemented";
//	};
//};

//class Direct : public ComputeMethod {
//public:
	void Direct::computeAccelerations(vector<body>& bodies) {
		const float G = 1.184E-4;// 6.67408 * pow(10, -11);
		int N = bodies.size();
		for (int i = 0; i < N; i++) {
			body& bi = bodies[i];
			for (int j = i + 1; j < N; j++) {
				body& bj = bodies[j];

				float dx = bj.x - bi.x;
				float dy = bj.y - bi.y;
				float r = sqrtf(powf(dx, 2) + powf(dy, 2));
				float ki = G * bj.mass / powf(r, 3);
				float kj = G * bi.mass / powf(r, 3);

				bi.ax += ki * dx;
				bi.ay += ki * dy;
				bj.ax -= kj * dx;
				bj.ay -= kj * dy;
			}
		}
	}
//};

//class DirectMultiThread : public ComputeMethod {
//public:
	void DirectMultiThread::computeAccelerations(vector<body>& bodies) {
		throw "Not implemented";
	}
//};
