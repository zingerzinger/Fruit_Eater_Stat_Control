#ifndef CREATURE_H
#define CREATURE_H

#include "vec2.h"

#define W_W 1280
#define W_H 1024

class Creature
{
public:
    Creature();

    void step(double dt_secs);

    void showFruit(int x, int y);

    void manualControl(double speed, double direction);

    Vec2 pos;
    Vec2 dir;
    double angle = 0.0;

    float rotSpeedDegs = 0.0f;
    float speedMps = 0.0f;
};

Creature::Creature()
{
    pos = Vec2(W_W / 2, W_H / 2);
}

void Creature::manualControl(double speed, double direction)
{
    speedMps = speed;
    rotSpeedDegs = direction;
}

void Creature::step(double dt_secs)
{

}

#endif // CREATURE_H
