#pragma once

#include "cuda_runtime.h"

class NbodySystem;

namespace IntegrationMethods {

	class IntegrationMethod {
	protected:
		NbodySystem* system;
	public:

		void setSystem(NbodySystem* system);
		virtual void integrate(float dt) = 0;
	};



	class Euler : public IntegrationMethod {
	public:
		//Euler();

		void integrate (float dt);
	};


	/*
	
	class EulerSymplectic : public IntegrationMethod {
	public:
		void integrate();
	};
	
	
	
	class VelocityVerlet : public IntegrationMethod {
	public:
		void integrate();
	};
	
	
	
	class ForestRuth : public IntegrationMethod {
	public:
		void integrate();
	};
	
	
	
	class PEFRL : public IntegrationMethod {
	public:
		void integrate();
	};
	
	*/
}