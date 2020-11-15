#pragma once

struct body
{
    double mass;
    double x, y;
    double Vx, Vy;
    double ax, ay;
    bool collided = false;
};