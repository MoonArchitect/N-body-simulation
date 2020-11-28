#pragma once

#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include "body.h"

namespace ComputeMethods {
	// bodies 
	class ComputeMethod {
	public:
		virtual void computeAccelerations(std::vector<body>& bodies) = 0;
		virtual void init(std::vector<body>& bodies) {};
		virtual void close() {};
	};

	class Direct : public ComputeMethod {
	public:
		void computeAccelerations(std::vector<body>& bodies) override;
	};
}
