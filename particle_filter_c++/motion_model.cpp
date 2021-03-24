#include <iostream>
#include <cmath>
#include <random>

using namespace std;

class MotionModel {
    /* References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics.
    MIT press, 2005. [Chapter 5] */

    public:
        float _alpha1, _alpha2, _alpha3, _alpha4;

        // Attributes of motion model class
        MotionModel(const float alpha1, const float alpha2, const float alpha3, const float alpha4) {
            _alpha1 = alpha1;
            _alpha2 = alpha2;
            _alpha3 = alpha3;
            _alpha4 = alpha4;
        }

        // Function this class can access
        void update(float *u_t0, float *u_t1, float *x_t0);
};

// Class function definition
void MotionModel::update(float *u_t0, float *u_t1, float *x_t0) {
    float x_diff, y_diff, th_diff, drot1, dtrans, drot2, drh1, dth, drh2;
    x_diff = u_t1[0] - u_t0[0];
    y_diff = u_t1[1] - u_t0[1];
    th_diff = u_t1[2] - u_t0[2];
    
    drot1 = atan2(y_diff, x_diff) - u_t0[2];
    dtrans = sqrt(pow(x_diff, 2.0) + pow(y_diff, 2.0));
    drot2 = u_t1[2] - u_t0[2] - drot1;

    default_random_engine generator;
    normal_distribution<float> sample1(0, sqrt((_alpha1 * drot1 * drot1) + (_alpha2 * dtrans * dtrans)));
    normal_distribution<float> sample2(0, sqrt((_alpha3 * dtrans * dtrans) + (_alpha4 * drot1 * drot1) + (_alpha4 * drot2 * drot2)));
    normal_distribution<float> sample3(0, sqrt((_alpha1 * drot2 * drot2) + (_alpha2 * dtrans * dtrans)));
    drh1 = drot1 - sample1(generator);
    dth = dtrans - sample2(generator);
    drh2 = drot2 - sample3(generator);

    x_t0[0] += dth * cos(x_t0[2] + drh1);
    x_t0[1] += dth * sin(x_t0[2] + drh1);
    x_t0[2] += drh1 + drh2;
}

// Main function (For testing purposes while other models are being written)
int main(){
    MotionModel particle1(0.1, 0.1, 0.1, 0.1);
    float x0[3] = { 400, 450, 0.785 };
    float u0[3] = { -94.234001, -139.953995, -1.342158 };
    float u1[3] = { -82.295998, -186.436005, -1.272345 };

    particle1.update(&u0[0], &u1[0], &x0[0]);
    cout << x0[0] << "," << x0[1] << "," << x0[2];

    return 0;
}