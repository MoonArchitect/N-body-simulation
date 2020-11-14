#pragma once

#include <vector>
#include "body.h"

using namespace std;

class ComputeMethod {
public:
	virtual void computeAccelerations(vector<body>& bodies) = 0;
};

class Direct : public ComputeMethod {
public:
	void computeAccelerations(vector<body>& bodies) override;
};

class DirectMultiThread : public ComputeMethod {
public:
	void computeAccelerations(vector<body>& bodies) override;
};
