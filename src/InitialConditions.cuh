#pragma once
#include "cuda_runtime.h"

class NbodySystem;


namespace InitialConditions {


	class InitialConditionsInterface {
	public:
		virtual void initialize(int offset, int n, NbodySystem* system) = 0;
	};


	class UniformBox : public InitialConditionsInterface {
		float2 pos, vel, size, massRange, velRange;
	public:
		UniformBox(float2 pos, float2 vel, float2 size, float2 massRange, float2 velRange);

		void initialize(int offset, int n, NbodySystem* system);
	};


}

