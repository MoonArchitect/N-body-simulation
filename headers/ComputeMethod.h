#pragma once

#include <vector>
#include "body.h"

namespace ComputeMethods {

	class ComputeMethod {
	public:
		virtual void computeAccelerations(std::vector<body>& bodies) = 0;
	};

	class Direct : public ComputeMethod {
	public:
		void computeAccelerations(std::vector<body>& bodies) override;
	};

	class DirectMultiThread : public ComputeMethod {
	public:
		void computeAccelerations(std::vector<body>& bodies) override;
	};

}