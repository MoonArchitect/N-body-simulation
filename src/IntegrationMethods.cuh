#pragma once

#include "cuda_runtime.h"

class NbodySystem;

namespace IntegrationMethods {

	class IntegrationMethod {
	protected:
		NbodySystem* system;
	public:

		virtual void setSystem(NbodySystem* system);
		virtual void integrate(float dt) = 0;
	};



	class Euler_Symplectic_KD : public IntegrationMethod {
	public:
		void integrate (float dt);
	};



	class Euler_Symplectic_DK : public IntegrationMethod {
	public:
		void integrate(float dt);
	};



	class Verlet_DKD : public IntegrationMethod {
	public:
		void integrate(float dt);
	};



	class Verlet_KDK : public IntegrationMethod {
		bool firstCalc = true;
	public:
		void setSystem(NbodySystem* system) override;

		void integrate(float dt);
	};
	


	class Velocity_Verlet : public IntegrationMethod {
		bool firstCalc = true;
		float4* d_prev_acc;
	public:
		void setSystem(NbodySystem* system) override;

		void integrate(float dt);
	};

	
	
	class ForestRuth : public IntegrationMethod {
	public:
		void integrate(float dt);
	};
	
	
	

	class PEFRL : public IntegrationMethod {
	public:
		void integrate(float dt);
	};
}
