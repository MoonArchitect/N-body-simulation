#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

struct body
{
    int mass;
    float x, y;
    float Vx, Vy;
    float ax, ay;
};

float systemEnergy(vector<body>& bodies) {
    const float G = 6.67408 * pow(10.f, -11.f);
    float Ek = 0, Ep = 0;
    for (int i = 0; i < bodies.size(); i++) {
        body& bi = bodies[i];
        Ek += bi.mass * (bi.Vx * bi.Vx + bi.Vy * bi.Vy) / 2.0f;
    }
    for (int i = 0; i < bodies.size(); i++) {
        body& bi = bodies[i];
        for (int j = 0; j < bodies.size(); j++) {
            if (j == i)
                continue;
            body& bj = bodies[j];
            float r = sqrt(pow(bj.x - bi.x, 2) + pow(bj.y - bi.y, 2));
            Ep += -G * bi.mass * bj.mass / 2.0f / r;
        }
    }

    return Ek + Ep;
}

void updateAcc(vector<body>& bodies, int N) {
    for (int i = 0; i < N; i++) {
        bodies[i].ax = 0;
        bodies[i].ay = 0;
    }
    for (int i = 0; i < N; i++) {
        body& bi = bodies[i];
        for (int j = i + 1; j < N; j++) {
            body& bj = bodies[j];
            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float r = sqrtf(powf(dx, 2) + powf(dy, 2));
            float ki = bj.mass / powf(r, 3);
            bi.ax += ki * dx;
            bi.ay += ki * dy;
            float kj = bi.mass / powf(r, 3);
            bj.ax -= kj * dx;
            bj.ay -= kj * dy;
        }
    }
}

void updatePos(vector<body>& bodies, float dt) {
    for (body& i : bodies) {
        i.Vx += i.ax * dt;
        i.Vy += i.ay * dt;

        i.x += i.Vx * dt;
        i.y += i.Vy * dt;
    }
}

void bodyRandomInit(vector<body>& bodies, float distanceRange[], float massRange[], float velocityRange[]) {
    for (body& i : bodies)
    {
        i.mass = (rand() / (float)RAND_MAX) * (massRange[1] - massRange[0]) + massRange[0];

        i.x = (rand() / (float)RAND_MAX) * (distanceRange[1] - distanceRange[0]) + distanceRange[0];
        i.y = (rand() / (float)RAND_MAX) * (distanceRange[1] - distanceRange[0]) + distanceRange[0];

        i.Vx = (rand() / (float)RAND_MAX) * (velocityRange[1] - velocityRange[0]) + velocityRange[0];
        i.Vy = (rand() / (float)RAND_MAX) * (velocityRange[1] - velocityRange[0]) + velocityRange[0];
    }
}

void writeTick(ofstream& data, vector<body>& bodies) {
    data << '[';
    for (body& i : bodies)
    {
        data << '[' << i.x << ',' << i.y << "],"; // [1,1]
    }
    data << "]," << '\n';
}

int main()
{
    const float G = 6.67408 * pow(10.f, -11.f);

    const int N = 500;
    const int cycles = 500;
    const float dt = 0.05; // s
    const float KmPerPixel = 1;
    vector<body> bodies(N);
    bodyRandomInit(
        bodies, 
        new float[2] { 300, 1500 },      // m
        new float[2] { 1000, 10000 },    // kg
        new float[2] { -2, 1.5 }         // m/s
    );

    ofstream data ("data.js");
    data << "var data = [";
    
    auto t1 = chrono::high_resolution_clock::now();

    for (int cycle = 0; cycle < cycles; cycle++) {
        updateAcc(bodies, N);
        updatePos(bodies, dt);
        if(cycle % 3 == 0)
            writeTick(data, bodies);
    }

    auto t2 = chrono::high_resolution_clock::now();
    data << "];";

    cout << (cycles * N) / (chrono::duration<double>(t2 - t1).count()) << "  obj*tick/sec \n";
}

