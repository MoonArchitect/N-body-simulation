#pragma once

class TimeStep {
public:
	virtual float getTimeStep() = 0;
};

class Constant : public TimeStep {
	float dt;
public:
	Constant(float dt);
	float getTimeStep() override;
};