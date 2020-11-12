#include "SystemInitializer.h"

using namespace std;


SystemInitializer::SystemInitializer(int N) : N(N) {}

float SystemInitializer::randomValue(float min, float max) {
	return ((float)rand() / RAND_MAX) * (max - min) + min;
}

int SystemInitializer::getN() { return this->N; }

void SystemInitializer::init(vector<body> &bodies) { // int startIdx = -1, int endIdx = -1
	throw "Initialization is not implemented";
}

// 
Random::Random(int N, float centerX, float centerY, float radius,
		float velocityRange, float massMin, float massMax,
		float centerVx, float centerVy)
		: SystemInitializer(N), 
		centerX(centerX), centerY(centerY), radius(radius),
		velocityRange(velocityRange), massMin(massMin), massMax(massMax),
		centerVx(centerVx), centerVy(centerVy) {}

void Random::init(vector<body>& bodies) {
	for (body& i : bodies) {
		i.mass = randomValue(this->massMin, this->massMax);
		i.x = randomValue(this->centerX - this->radius, this->centerX + this->radius);
		i.y = randomValue(this->centerY - this->radius, this->centerY + this->radius);
		i.Vx = randomValue(-this->velocityRange, this->velocityRange);
		i.Vy = randomValue(-this->velocityRange, this->velocityRange);
	}
}


