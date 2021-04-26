#pragma once
#include "cuda_runtime.h"

class NbodySystem;

namespace ComputeMethods {
	
	class ComputeMethod {

	protected:
		const float SOFTENING;
		NbodySystem *system;
	public:
		int nodes = -1;

		ComputeMethod(const float softening);
		
		void setSystem(NbodySystem* system);
		virtual void computeAcc() = 0;
	};

	
	class Direct : public ComputeMethod {
	public:
		Direct(const float softening);

		void computeAcc();
	};


	class BarnesHut : public ComputeMethod {
		float4* d_bounds;
	public:
		BarnesHut(int nodes, const float softening);
		
		void computeAcc();
	};

}