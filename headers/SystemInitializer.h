#pragma once

#include "body.h"
#include <vector>

using namespace std;
namespace SystemInitializers {

	class SystemInitializer {
	protected:
		int N;
		SystemInitializer(int N);
		float randomValue(float min, float max);

	public:
		int getN();
		virtual void init(vector<body>& bodies);
	};

	class Random : public SystemInitializer {
		float centerX, centerY, radius, centerVx, centerVy, velocityRange, massMin, massMax;
	public:
		Random(int N, float centerX, float centerY, float radius,
			float velocityRange, float massMin, float massMax,
			float centerVx = 0, float centerVy = 0);

		void init(vector<body>& bodies) override;
	};

	class StarSystem : public SystemInitializer {
		// x, y, velocity center
		// # suns, # planets, # moons
		// sun masses, planet masses
		float stars, planets, moons, centerX, centerY, radius, centerVx, centerVy, stability;
	public:
		StarSystem(int N, float centerX, float centerY, float radius,
			float stability, float stars = 0, float planets = 0, float moons = 0, float centerVx = 0, float centerVy = 0);

		void init(vector<body>& bodies) override;
	};

	class GlobularCluster : public SystemInitializer {
		// x, y, velocity center
		// r, N
	};

	class EllipticalGalaxy : public SystemInitializer {
		// x, y, velocity center
		// r, N obj
	};

	class SpiralGalaxy : public SystemInitializer {
		// x, y, velocity center
		// r, N obj, n spirals
	};
}