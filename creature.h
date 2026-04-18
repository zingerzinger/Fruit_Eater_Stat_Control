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

    double w = 0.0;
    double orientation = 0.0;

    double vel = 0.0;
    Vec2 pos;

    float rotF = 0.0f;
    float fwdF = 0.0f;

    QQueue<Vec2> trail;

    QVector<Vec2> fruits;

    Vec2 target;
    double angleError = 0.0;
    double targetDistance = 0.0;
    bool targetChange = false;

    bool manual = true;

private:

    Vec2 lpos;
    uint64_t ltimeNoFruit = 0;

    int targetIdx = -1;
    int prevTargetIdx = -1;

    double deltaDegsPrev = 0.0;

    int targetCounter = 10;

    uint64_t frameNum     = 0;
    uint64_t frameNumPrev = 0;
    double targetDistancePrev = 0;

    void calcWeights();

    void control_PD(double w,
                    double orientation,
                    Vec2   pos,
                    double vel,
                    float  rotF,
                    float  fwdF,
                    Vec2   targetPos,
                    double angleError,
                    double angleErrorDelta,
                    double targetDistance);
};

Creature::Creature()
{
    pos = Vec2(W_W / 2, W_H / 2);
}

void Creature::manualControl(double speed, double direction)
{
    if (!manual) { return; }
    fwdF = speed;
    rotF = direction;
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

// === === ===

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

    frameNum++;

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
        targetIdx = fidx;
    } else { // no fruits, choose random target position to roam

        uint64_t ctime = micros();

        if (/*ctime - ltimeNoFruit >= CREATURE_NO_FRUIT_RND_USEC || */
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

            targetCounter++;
            targetIdx = targetCounter;
        }
    }

    if (pos.x < 0 || pos.x > W_W ||
        pos.y < 0 || pos.y > W_H) { pos = Vec2(W_W * 0.5, W_H * 0.5); }

    targetChange = targetIdx != prevTargetIdx;
    prevTargetIdx = targetIdx;

    targetDistance = VecLen( subVec(target, pos) );

    Vec2 orientRay = VecUnit( rotateDegs(orientation, Vec2(1, 0)) );
    Vec2 targetRay = VecUnit( subVec(target, pos) );

    // l : +, r : -
    double deltaDegs = rad2deg( asin(vecCross(orientRay, targetRay)) );

    // === === ===

    if (targetChange) {
        frameNumPrev = frameNum;
        targetDistancePrev = targetDistance;
    }

    // === === ===

    control_PD(w,
               orientation,
               pos,
               vel,
               rotF,
               fwdF,
               target,
               deltaDegs,
               deltaDegs - deltaDegsPrev,
               targetDistance);

    deltaDegsPrev = deltaDegs;
}

void Creature::control_PD(double in_w,
                          double in_orientation,
                          Vec2   in_pos,
                          double in_vel,
                          float  in_rotF,
                          float  in_fwdF,
                          Vec2   in_targetPos,
                          double in_angleError,
                          double angleErrorDelta,
                          double in_targetDistance)
{
    double dir = in_angleError > 0 ? 1 : -1;

    rotF = abs(in_angleError) * dir;

    if (abs(in_angleError) < CREATURE_DELTA_ANGLE_ROT_DEGS) {
        fwdF = 1;
    }

    rotF = (CREATURE_ROT_P_K * abs(in_angleError)) * dir + CREATURE_ROT_D_K * angleErrorDelta;
}

#endif // CREATURE_H
