#include "NbodySystem.h"

#define BLOCK_SIZE 128

using namespace IntegrationMethods;

void IntegrationMethod::setSystem(NbodySystem* system) {
	this->system = system;
}

/////////////////////////////////  Symplectic Euler (Kick-Drift variant) /////////////////////////////////

__global__ void KD_Symplectic_Euler(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * acc[i].x;
		vel[i].y += dt * acc[i].y;
		vel[i].z += dt * acc[i].z;

		pos[i].x += vel[i].x * dt;
		pos[i].y += vel[i].y * dt;
		pos[i].z += vel[i].z * dt;
	}
}

	
void Euler_Symplectic_KD::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	system->computeAcceleration();

	KD_Symplectic_Euler <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	cudaDeviceSynchronize();
}


/////////////////////////////////  Symplectic Euler (Drift-Kick variant) /////////////////////////////////

__global__ void DK_Symplectic_Euler_1(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		pos[i].x += vel[i].x * dt;
		pos[i].y += vel[i].y * dt;
		pos[i].z += vel[i].z * dt;
	}
}

__global__ void DK_Symplectic_Euler_2(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * acc[i].x;
		vel[i].y += dt * acc[i].y;
		vel[i].z += dt * acc[i].z;
	}
}


void Euler_Symplectic_DK::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	DK_Symplectic_Euler_1 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	
	system->computeAcceleration();

	DK_Symplectic_Euler_2 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	cudaDeviceSynchronize();
}


////////////////////////////////  Verlet Method (Drift-Kick-Drift variant) ///////////////////////////////

__global__ void DKD_Verlet_1(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		pos[i].x += vel[i].x * (dt * 0.5);
		pos[i].y += vel[i].y * (dt * 0.5);
		pos[i].z += vel[i].z * (dt * 0.5);
	}
}

__global__ void DKD_Verlet_2(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * acc[i].x;
		vel[i].y += dt * acc[i].y;
		vel[i].z += dt * acc[i].z;

		pos[i].x += vel[i].x * (dt * 0.5);
		pos[i].y += vel[i].y * (dt * 0.5);
		pos[i].z += vel[i].z * (dt * 0.5);
	}
}


void Verlet_DKD::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	DKD_Verlet_1 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	system->computeAcceleration();

	DKD_Verlet_2 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	cudaDeviceSynchronize();
}


////////////////////////////////  Verlet Method (Kick-Drift-Kick variant) ////////////////////////////////

__global__ void KDK_Verlet_1(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * 0.5 * acc[i].x;
		vel[i].y += dt * 0.5 * acc[i].y;
		vel[i].z += dt * 0.5 * acc[i].z;

		pos[i].x += vel[i].x * dt;
		pos[i].y += vel[i].y * dt;
		pos[i].z += vel[i].z * dt;
	}
}

__global__ void KDK_Verlet_2(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += dt * 0.5 * acc[i].x;
		vel[i].y += dt * 0.5 * acc[i].y;
		vel[i].z += dt * 0.5 * acc[i].z;
	}
}


void Verlet_KDK::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (firstCalc) {
		firstCalc = false;
		system->computeAcceleration();
	}

	KDK_Verlet_1 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	
	system->computeAcceleration();
	
	KDK_Verlet_2 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	cudaDeviceSynchronize();
}


//////////////////////////////////////////    Velocity Verlet   //////////////////////////////////////////

__global__ void Velocity_Verlet_1(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		pos[i].x += vel[i].x * dt + acc[i].x * 0.5 * dt * dt;
		pos[i].y += vel[i].y * dt + acc[i].y * 0.5 * dt * dt;
		pos[i].z += vel[i].z * dt + acc[i].z * 0.5 * dt * dt;
	}
}

__global__ void Velocity_Verlet_2(float4* pos, float4* vel, float4* acc, float4* prev_acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		vel[i].x += 0.5 * dt * (acc[i].x + prev_acc[i].x);
		vel[i].y += 0.5 * dt * (acc[i].y + prev_acc[i].y);
		vel[i].z += 0.5 * dt * (acc[i].z + prev_acc[i].z);
	}
}


void Velocity_Verlet::setSystem(NbodySystem* system) {
	IntegrationMethod::setSystem(system);

	cudaMalloc(&d_prev_acc, system->N * sizeof(float4)); 
	cudaMemset(d_prev_acc, 0, system->N * sizeof(float4));
}

void Velocity_Verlet::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (firstCalc) {
		firstCalc = false;
		system->computeAcceleration();
	}

	Velocity_Verlet_1 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	std::swap(d_prev_acc, system->device.acc);

	system->computeAcceleration();

	Velocity_Verlet_2 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, d_prev_acc, dt, system->N);
	cudaDeviceSynchronize();
}


///////////////////////////////////////////     ForestRuth    ////////////////////////////////////////////

__global__ void ForestRuth_1(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	const double O = 1.35120719195966;

	if (i < n) {
		pos[i].x += vel[i].x * dt * O * 0.5;
		pos[i].y += vel[i].y * dt * O * 0.5;
		pos[i].z += vel[i].z * dt * O * 0.5;
	}
}

__global__ void ForestRuth_2(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	const double O = 1.35120719195966;

	if (i < n) {
		vel[i].x += dt * acc[i].x * O;
		vel[i].y += dt * acc[i].y * O;
		vel[i].z += dt * acc[i].z * O;

		pos[i].x += vel[i].x * dt * (1 - O) * 0.5;
		pos[i].y += vel[i].y * dt * (1 - O) * 0.5;
		pos[i].z += vel[i].z * dt * (1 - O) * 0.5;
	}
}

__global__ void ForestRuth_3(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	const double O = 1.35120719195966;

	if (i < n) {
		vel[i].x += dt * acc[i].x * (1 - 2 * O);
		vel[i].y += dt * acc[i].y * (1 - 2 * O);
		vel[i].z += dt * acc[i].z * (1 - 2 * O);

		pos[i].x += vel[i].x * dt * (1 - O) * 0.5;
		pos[i].y += vel[i].y * dt * (1 - O) * 0.5;
		pos[i].z += vel[i].z * dt * (1 - O) * 0.5;
	}
}

__global__ void ForestRuth_4(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	const double O = 1.35120719195966;

	if (i < n) {
		vel[i].x += dt * acc[i].x * O;
		vel[i].y += dt * acc[i].y * O;
		vel[i].z += dt * acc[i].z * O;

		pos[i].x += vel[i].x * dt * O * 0.5;
		pos[i].y += vel[i].y * dt * O * 0.5;
		pos[i].z += vel[i].z * dt * O * 0.5;
	}
}


void ForestRuth::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	ForestRuth_1 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	
	system->computeAcceleration();

	ForestRuth_2 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	
	system->computeAcceleration();

	ForestRuth_3 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	
	system->computeAcceleration();

	ForestRuth_4 <<< nBlocks, BLOCK_SIZE >>> (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);
	cudaDeviceSynchronize();
}


//////////////////////////////////////////////    PERFL    ///////////////////////////////////////////////

__global__ void PEFRL_1(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double Et = 0.1786178958448091 * dt;

	if (i < n) {
		pos[i].x += Et * vel[i].x;
		pos[i].y += Et * vel[i].y;
		pos[i].z += Et * vel[i].z;
	}
}

__global__ void PEFRL_2(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double mL = (1 + 2 * 0.2123418310626054) / 2 * dt;
	double Xt = -0.06626458266981849 * dt;

	if (i < n) {
		vel[i].x += mL * acc[i].x;
		vel[i].y += mL * acc[i].y;
		vel[i].z += mL * acc[i].z;

		pos[i].x += Xt * vel[i].x;
		pos[i].y += Xt * vel[i].y;
		pos[i].z += Xt * vel[i].z;
	}
}

__global__ void PEFRL_3(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double Lt = -0.2123418310626054 * dt;
	double kt = (1 - 2 * (0.1786178958448091 - 0.06626458266981849)) * dt;

	if (i < n) {
		vel[i].x += Lt * acc[i].x;
		vel[i].y += Lt * acc[i].y;
		vel[i].z += Lt * acc[i].z;

		pos[i].x += kt * vel[i].x;
		pos[i].y += kt * vel[i].y;
		pos[i].z += kt * vel[i].z;
	}
}

__global__ void PEFRL_4(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double Lt = -0.2123418310626054 * dt;
	double Xt = -0.06626458266981849 * dt;

	if (i < n) {
		vel[i].x += Lt * acc[i].x;
		vel[i].y += Lt * acc[i].y;
		vel[i].z += Lt * acc[i].z;

		pos[i].x += Xt * vel[i].x;
		pos[i].y += Xt * vel[i].y;
		pos[i].z += Xt * vel[i].z;
	}
}

__global__ void PEFRL_5(float4* pos, float4* vel, float4* acc, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double mL = (1 + 2 * 0.2123418310626054) / 2 * dt;
	double Et = 0.1786178958448091 * dt;

	if (i < n) {
		vel[i].x += mL * acc[i].x;
		vel[i].y += mL * acc[i].y;
		vel[i].z += mL * acc[i].z;

		pos[i].x += Et * vel[i].x;
		pos[i].y += Et * vel[i].y;
		pos[i].z += Et * vel[i].z;
	}
}


void PEFRL::integrate(float dt) {
	int nBlocks = (system->N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	PEFRL_1 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	system->computeAcceleration();

	PEFRL_2 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	system->computeAcceleration();
	
	PEFRL_3 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	system->computeAcceleration();

	PEFRL_4 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	system->computeAcceleration();

	PEFRL_5 << < nBlocks, BLOCK_SIZE >> > (system->device.pos_mass, system->device.vel, system->device.acc, dt, system->N);

	cudaDeviceSynchronize();
}


