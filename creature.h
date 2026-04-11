#ifndef CREATURE_H
#define CREATURE_H

#include <QQueue>

#include "defines.h"
#include "utils.h"

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

    QVector<Vec2> fruits;
    Vec2 target;
    bool manual = true;


private:

    Vec2 lpos;
    uint64_t ltimeNoFruit = 0;
};

Creature::Creature()
{
    pos = Vec2(W_W / 2, W_H / 2);
}

void Creature::manualControl(double speed, double direction)
{
    if (!manual) { return; }
    speedMps = speed;
    rotSpeedDegs = direction;
}

void Creature::showFruit(Vec2 f)
{
    if (fruits.size() >= CREATURE_MEM_MAX) { return; }

    for (Vec2 fm : fruits) {
        if (fm.x - FRUIT_MEM_MARGIN <= f.x &&
            fm.x + FRUIT_MEM_MARGIN >= f.x &&
            fm.y - FRUIT_MEM_MARGIN <= f.y &&
            fm.y + FRUIT_MEM_MARGIN >= f.y) { return; }
    }

    fruits.append(f);
}

void Creature::removeFruit(Vec2 f)
{
    int i = 0;
    for (Vec2 fm : fruits) {

        if (fm.x - FRUIT_MEM_MARGIN <= f.x &&
            fm.x + FRUIT_MEM_MARGIN >= f.x &&
            fm.y - FRUIT_MEM_MARGIN <= f.y &&
            fm.y + FRUIT_MEM_MARGIN >= f.y) {

            fruits.remove(i);
            return;
        }

        i++;
    }
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

    if (manual) { return; }

    // target calculation - search for closest ETA fruit ((?)rotation + distance)

    double minTime = MAXFLOAT;
    int fidx = -1;
    int i = 0;

    for (Vec2 f : fruits) {
        double dist = VecLenSq( subVec(f, pos) );
        if (dist < minTime) { minTime = dist; fidx = i; }
        i++;
    }

    if (fidx >= 0) { // found the target
        target = fruits[fidx];
    } else { // no fruits, choose random target position to roam

        uint64_t ctime = micros();

        if (ctime - ltimeNoFruit >= CREATURE_NO_FRUIT_RND_USEC ||
            VecLenSq( subVec(target, pos) ) <= /* do not get too close to the random target to counter singularity */
            (CREATURE_RND_MIN_TARGET_APPROACH_DIST * CREATURE_RND_MIN_TARGET_APPROACH_DIST)) {

            ltimeNoFruit = ctime;

            int attempts = 0;

            while (true) {
                target = Vec2(fRand(0, W_W), fRand(0, W_H));

                if ( VecLenSq( subVec(target, pos) ) >= CREATURE_RND_MIN_TARGET_DIST) { break; }

                attempts++;
                if (attempts >= CREATURE_RND_TARGET_ATTEMPTS) {
                    target = Vec2(); // goto left top corner
                    break;
                }
            }
        }
    }

    // target

    Vec2 orientRay = VecUnit( rotateDegs(angle, Vec2(1, 0)) );
    Vec2 targetRay = VecUnit( subVec(target, pos) );

    //double tdistSq = VecLenSq( subVec(target, pos) );

    // l : +, r : -
    double deltaDegs = rad2deg( asin(vecCross(orientRay, targetRay)) );

    double dir = deltaDegs > 0 ? 1 : -1;
    rotSpeedDegs = abs(deltaDegs) * dir;

    if (abs(deltaDegs) < CREATURE_DELTA_ANGLE_ROT_DEGS) {
        speedMps = CREATURE_MAX_SPEED;
        rotSpeedDegs = (CREATURE_ROT_P_K * abs(deltaDegs)) * dir;
    }

    rotSpeedDegs = clamp(rotSpeedDegs, -CREATURE_MAX_ROT_SPEED_DPS, CREATURE_MAX_ROT_SPEED_DPS);
}

#endif // CREATURE_H
