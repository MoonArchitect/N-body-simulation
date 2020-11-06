#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>

using namespace std;

struct body
{
    int mass;
    float x, y;
    float Vx, Vy;
    float ax, ay;
    int collided = -1;
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

void updateAcc2(vector<body>& bodies, int idx = 0, int total = 1) {
    const float G = 6.67408 * pow(10, -11);
    int N = bodies.size();
    for (int i = 0; i < N; i++) {
        body& bi = bodies[i];
        for (int j = i + 1; j < N; j++) {
            body& bj = bodies[j];

            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float r = sqrtf(powf(dx, 2) + powf(dy, 2));
            float ki = G * bj.mass / powf(r, 3);
            float kj = G * bi.mass / powf(r, 3);

            bi.ax += ki * dx;
            bi.ay += ki * dy;
            bj.ax -= kj * dx;
            bj.ay -= kj * dy;
        }
    }
}


void updatePos(vector<body>& bodies, float dt) {
    const float G = 6.67408 * pow(10, -11);
    int N = bodies.size();
    for (int i = 0; i < N; i++) {
        body& bi = bodies[i];
        bi.x += bi.Vx * dt;
        bi.y += bi.Vy * dt;
    }

    for (int i = 0; i < N; i++) {
        body& bi = bodies[i];
        for (int j = i + 1; j < N; j++) {
            body& bj = bodies[j];

            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float r = sqrtf(powf(dx, 2) + powf(dy, 2));
            float ki = G * bj.mass / powf(r, 3);
            float kj = G * bi.mass / powf(r, 3);

            bi.ax += ki * dx;
            bi.ay += ki * dy;
            bj.ax -= kj * dx;
            bj.ay -= kj * dy;
        }
    }

    for (int i = 0; i < N; i++) {
        body& bi = bodies[i];
        bi.Vx += bi.ax * dt;
        bi.Vy += bi.ay * dt;
    }


    for (int i = 0; i < N; i++) {
        body& bi = bodies[i];
        for (int j = 0; j < N; j++) {
            if (j == i || bi.collided != -1 || bodies[j].collided != -1)
                continue;
    
            body& bj = bodies[j];
    
            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float r = sqrtf(powf(dx, 2) + powf(dy, 2));
    
            if (r < 10.0f) {
                if (bj.mass > bi.mass) {
                    bi.collided = 1;
                    //cout << bi.mass << "  " << bj.mass << "  " << (bi.Vx * bi.mass + bj.Vx * bj.mass) << '\n';
                    bj.Vx = (bi.Vx * bi.mass + bj.Vx * bj.mass) / (bj.mass + bi.mass);
                    bj.Vy = (bi.Vy * bi.mass + bj.Vy * bj.mass) / (bj.mass + bi.mass);
                    bj.mass += bi.mass;
                }
                else {
                    bj.collided = 1;
                    //cout << bi.mass << "  " << bj.mass << "  " << (bi.Vx * bi.mass + bj.Vx * bj.mass) << '\n';
                    bi.Vx = (bi.Vx * bi.mass + bj.Vx * bj.mass) / (bj.mass + bi.mass);
                    bi.Vy = (bi.Vy * bi.mass + bj.Vy * bj.mass) / (bj.mass + bi.mass);
                    bi.mass += bj.mass;
                }
            }
        }
    }
    
    for (int i = 0; i < bodies.size(); i++) {
        if (bodies[i].collided == 1) {
            bodies.erase(bodies.begin() + i);
            i--;
        }
    }
    
}

void bodyRandomInit(vector<body>& bodies, float distanceRange[], float massRange[], float velocityRange[]) {
    for (body& i : bodies)
    {
        i.mass = (rand() / (float)RAND_MAX) * (massRange[1] - massRange[0]) + massRange[0];

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        normal_distribution<float> distribution(1600.0, 800.0);

        i.x = distribution(generator);
        i.y = distribution(generator);

        i.Vx = (rand() / (float)RAND_MAX) * (velocityRange[1] - velocityRange[0]) + velocityRange[0];
        i.Vy = (rand() / (float)RAND_MAX) * (velocityRange[1] - velocityRange[0]) + velocityRange[0];
    }
}

void bodyStarSystemInit(vector<body>& bodies, float offset, float r, float starMass) {
    body& star = bodies[0];
    int N = bodies.size();
    star.mass = starMass;
    star.x = offset;
    star.y = offset;
    
    N--;
    for (int i = 1; i < (N * 0.3); i++) {
        body& bi = bodies[i];
        bi.mass = starMass;

        bi.x = (rand() / (float)RAND_MAX) * (r + r) + (offset - r);
        bi.y = (rand() / (float)RAND_MAX) * (r + r) + (offset - r);
    }

    for (int i = N * 0.3; i <= N; i++) {
        body& bi = bodies[i];
        bi.mass = starMass / 10000;

        bi.x = (rand() / (float)RAND_MAX) * (r + r) + (offset - r);
        bi.y = (rand() / (float)RAND_MAX) * (r + r) + (offset - r);
    }

    updateAcc2(bodies, 0, 1);

    for (int i = 1; i < bodies.size(); i++) {
        body& bi = bodies[i];
        float dx = bi.x - bodies[0].x;
        float dy = bi.y - bodies[0].y;
        float r = sqrt(pow(dx, 2) + pow(dy, 2));
        bi.Vx = -sqrt(abs(bi.ay * r)) * abs(bi.ay) / bi.ay * (rand() / (float)RAND_MAX / 3 + 0.6);
        bi.Vy = sqrt(abs(bi.ax * r)) * abs(bi.ax) / bi.ax * (rand() / (float)RAND_MAX / 3 + 0.6);
    }
}

void writeTick(ofstream& data, vector<body>& bodies) {
    data << '[';
    for (body& i : bodies)
    {
        data << '[' << i.x << ',' << i.y << ',' << i.mass << "],"; // [1,1]
    }
    data << "]," << '\n';
}

int main() {
    int a[] = { 16 };
    int r[] = { 400 };
    float EnergyCorrection = 0;
    for (int N : a) {
        for (int rad : r) {
            srand(2312223);
            cout << " ----------------->> " << N << " | " << rad << '\n';
            const int cycles = 2500000;
            const float dt = 0.7f;
            vector<body> bodies(N);
            
            bodyStarSystemInit(bodies, 2000, rad, 15000000);

            ofstream data("data.js");
            data << "var data = [";

            auto t1 = chrono::high_resolution_clock::now();
            float energy = systemEnergy(bodies);
            float totalE = 0;
            for (int cycle = 0; cycle < cycles; cycle++) {
                for (int i = 0; i < bodies.size(); i++) {
                    bodies[i].ax = 0;
                    bodies[i].ay = 0;
                }

                if (cycle % 25000 == 0)
                    std::cout << cycle << '\n';
                if (cycle % 5000 == 0) {
                    float current = systemEnergy(bodies);
                    totalE += current;
                    if (cycle % 25000 == 0)
                        std::cout << current / energy << "   |   " << totalE / ((int)(cycle / 500) + 1) / energy << '\n';
                }
                updatePos(bodies, dt);

                if (cycle % 500 == 0)
                    writeTick(data, bodies);
            }
            float current = systemEnergy(bodies);
            std::cout << totalE / ((int)(cycles / 50)) / energy << '\n';
            auto t2 = chrono::high_resolution_clock::now();
            data << "];";

            std::cout << (cycles) / (chrono::duration<float>(t2 - t1).count()) << "  tick/sec \n\n\n";
        }
    }
}

