#include "SystemInitializer.h"

using namespace std;
using namespace SystemInitializers;


SystemInitializer::SystemInitializer(int N) : N(N) { if (N < 2) { throw "Not eanough objects, N has to be larger than 1"; } }

float SystemInitializer::randomValue(float min, float max) {
	return ((float)rand() / RAND_MAX) * (max - min) + min;
}

int SystemInitializer::getN() { return N; }

//		change params to double

// -------------------------------  Random  -------------------------------
Random::Random(int N, float centerX, float centerY, float radius,
		float velocityRange, float massMin, float massMax,
		float centerVx, float centerVy)
		: SystemInitializer(N),
		centerX(centerX), centerY(centerY), radius(radius),
		velocityRange(velocityRange), massMin(massMin), massMax(massMax),
		centerVx(centerVx), centerVy(centerVy) {}

void Random::init(vector<body>& bodies, int startId, int endId) {
	//if (startIdx < 0 || startIdx > endIdx || endIdx > bodies.size())
	for (int id = startId; id < endId; id++) {
		body& i = bodies[id];
		i.mass = randomValue(massMin, massMax);
		i.x = randomValue(centerX - radius, centerX + radius);
		i.y = randomValue(centerY - radius, centerY + radius);
		i.Vx = randomValue(-velocityRange, velocityRange) + centerVx;
		i.Vy = randomValue(-velocityRange, velocityRange) + centerVy;
	}
}

// -------------------------------  Star System  -------------------------------
StarSystem::StarSystem(
		int N, float centerX, float centerY, float radius,
		float stability, float stars, float planets, float moons, 
		float centerVx, float centerVy
	): SystemInitializer(N), centerX(centerX), centerY(centerY), radius(radius),
	stability(stability), stars(stars), planets(planets), moons(moons), 
	centerVx(centerVx), centerVy(centerVy) {}

void StarSystem::init(vector<body>& bodies, int startId, int endId) {
		
}

// -------------------------------  GlobularCluster  -------------------------------


// -------------------------------  EllipticalGalaxy  -------------------------------


// -------------------------------  SpiralGalaxy  -------------------------------

