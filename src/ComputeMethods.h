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

		ComputeMethod(int nodes, const float softening);
		
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
		int bodiesPerBlock;
		int *d_index, *d_nodes, *d_validBodies, *d_validBodiesTop, *d_count, *d_idx_to_body, *d_start;
		float4 *d_bounds;
		float theta;

	public:
		BarnesHut(float theta, int nodes, const float softening);
		
		void setSystem(NbodySystem* system);
		void computeAcc();
	};

}