#pragma once
#include "cuda_runtime.h"

class NbodySystem;

namespace ComputeMethods {
	
	class ComputeMethod {

	protected:
		const float SOFTENING;
		NbodySystem *system;
	public:
		int knodes = 1;

		ComputeMethod(int knodes, const float softening);
		
		virtual void setSystem(NbodySystem* system) = 0;
		virtual void computeAcc() = 0;
	};

	
	class Direct : public ComputeMethod {
	public:
		Direct(const float softening);

		void setSystem(NbodySystem* system);
		void computeAcc();
	};


	class BarnesHut : public ComputeMethod {
		int *d_index, *d_nodes, *sortedIdx, *SFCkeys;
		float4 *d_bounds;
		float theta;

	public:
		BarnesHut(float theta, int knodes, const float softening);
		
		void setSystem(NbodySystem* system);
		void computeAcc();
	};

}