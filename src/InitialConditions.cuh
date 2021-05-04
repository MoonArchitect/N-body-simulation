#pragma once
#include "cuda_runtime.h"

class NbodySystem;


namespace InitialConditions {

	class InitialConditions {
	public:
		virtual void initialize(int offset, int n, NbodySystem* system) = 0;
	};

	namespace Modules {

		class UniformBox : public InitialConditions {
			float2 pos, vel, size, massRange, velRange;
		public:
			UniformBox(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange);

			void initialize(int offset, int n, NbodySystem* system);
		};


		class UniformEllipsoid : public InitialConditions {
		protected:
			float2 pos, vel, radius, massRange, velRange;
		public:
			UniformEllipsoid(float2 pos, float2 vel, float2 radius, float2 massRange, float2 velRange);

			void initialize(int offset, int n, NbodySystem* system);
		};
	}



	class DiskModel : public Modules::UniformEllipsoid {

	public:
		DiskModel(float2 pos, float2 vel, float2 radius, float2 massRange, float2 velRange);

		void initialize(int offset, int n, NbodySystem* system);
	};



	class Standard : public InitialConditions {
		float2 pos, vel, size, massRange, velRange;
	public:
		Standard(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange);

		void initialize(int offset, int n, NbodySystem* system);
	};
}

