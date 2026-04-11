#ifndef VEC_H
#define VEC_H

#include <chrono>
#include <atomic>
#include <mutex>
#include <math.h>
#include <random>

#define M_PI 3.14159265358979323846
#define EULER 2.71828182845904523536

struct Vec2 {

public:

	double x = 0.0;
	double y = 0.0;

	Vec2() : x(0), y(0) {

	}

	Vec2(double X, double Y) : x(X), y(Y) {
		
	}
};

Vec2 rotateDegs(double degs, Vec2 vi);

Vec2 addVec(Vec2 pa, Vec2 pb);
Vec2 subVec(Vec2 pa, Vec2 pb);

Vec2 mulVecScalar(double scalar, Vec2 pi);

double vecDot(Vec2 a, Vec2 b);

Vec2   VecUnit (Vec2 vi);
double VecLen  (Vec2 vi);
double VecLenSq(Vec2 vi);

double clamp(double value, double min, double max);

// =============================

Vec2 rotateDegs(double degs, Vec2 vi) {

    double rads = degs * M_PI / 180.0;

    double c = cos(rads);
    double s = sin(rads);

    Vec2 v;

    v.x = c * vi.x - s * vi.y;
    v.y = s * vi.x + c * vi.y;

    return v;
}

Vec2 addVec(Vec2 pa, Vec2 pb) {
    Vec2 r;
    r.x = pa.x + pb.x;
    r.y = pa.y + pb.y;
    return r;
}

Vec2 subVec(Vec2 pa, Vec2 pb) {
    Vec2 r;
    r.x = pa.x - pb.x;
    r.y = pa.y - pb.y;
    return r;
}

Vec2 mulVecScalar(double scalar, Vec2 pi) {
    Vec2 r;
    r.x = pi.x * scalar;
    r.y = pi.y * scalar;
    return r;
}

double vecDot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

double vecCross(Vec2 a, Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

Vec2 VecUnit(Vec2 vi) {
    Vec2 r;
    double l = VecLen(vi);
    r.x = vi.x / l;
    r.y = vi.y / l;
    return r;
}

double VecLen(Vec2 vi) {
    return sqrt(VecLenSq(vi));
}

double VecLenSq(Vec2 vi) {
    return vi.x * vi.x + vi.y * vi.y;
}

double clamp(double value, double min, double max) {
    if (value < min) { return min; }
    if (value > max) { return max; }
    return value;
}

double rad2deg(double rads) { return rads * 180.0f / M_PI; }

double deg2rad(double degs) { return degs * M_PI / 180.0f; }

uint64_t micros() {
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
    return us;
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

#endif // VEC_H
