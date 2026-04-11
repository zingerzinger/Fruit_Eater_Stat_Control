#ifndef CREATURE_H
#define CREATURE_H

#include <QQueue>

#include "defines.h"
#include "vec2.h"

class Creature
{
public:
    Creature();

    void step(double dt_secs);

    void showFruit(Vec2 f);
    void removeFruit(Vec2 f);

    void manualControl(double speed, double direction);

    Vec2 pos;
    Vec2 dir;
    double angle = 0.0;

    float rotSpeedDegs = 0.0f;
    float speedMps = 0.0f;

    QQueue<Vec2> trail;

private:

    Vec2 lpos;
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

void Creature::showFruit(Vec2 f)
{

}

void Creature::removeFruit(Vec2 f)
{

}

void Creature::step(double dt_secs)
{
    // trail
    double dsq = VecLenSq( subVec(pos, lpos) );
    if (dsq >= CREATURE_TRAJECTORY_MIN_STEP * CREATURE_TRAJECTORY_MIN_STEP) {
        trail.append(pos);
        lpos = pos;
    }
    if (trail.size() > CREATURE_TRAJECTORY_SIZE) { trail.removeFirst(); }
}

#endif // CREATURE_H
