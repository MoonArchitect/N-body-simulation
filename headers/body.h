#pragma once

struct body
{
    float mass;
    float x, y;
    float Vx, Vy;
    float ax, ay;
    bool collided = false;
};
