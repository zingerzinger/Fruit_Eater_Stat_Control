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
                    double targetDistance);

    void control_NN(double w,
                    double orientation,
                    Vec2   pos,
                    double vel,
                    float  rotF,
                    float  fwdF,
                    Vec2   targetPos,
                    double angleError,
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

            targetCounter++;
            targetIdx = targetCounter;
        }
    }

    targetChange = targetIdx != prevTargetIdx;
    prevTargetIdx = targetIdx;

    Vec2 orientRay = VecUnit( rotateDegs(orientation, Vec2(1, 0)) );
    Vec2 targetRay = VecUnit( subVec(target, pos) );

    // l : +, r : -
    double deltaDegs = rad2deg( asin(vecCross(orientRay, targetRay)) );

    // === === ===

    if (targetChange) {
        int64_t T = frameNum - frameNumPrev;
        double  D = targetDistancePrev;
        double  TTT = D / T;

        frameNumPrev = frameNum;
        targetDistancePrev = targetDistance;

        //qDebug() << TTT;
    }

    // === === ===

    control_NN(w,
               orientation,
               pos,
               vel,
               rotF,
               fwdF,
               target,
               deltaDegs,
               targetDistance);
}

void Creature::control_PD(double in_w,
                          double in_orientation,
                          Vec2   in_pos,
                          double in_vel,
                          float  in_rotF,
                          float  in_fwdF,
                          Vec2   in_targetPos,
                          double in_angleError,
                          double in_targetDistance)
{
    double angleErrorDelta = in_angleError - deltaDegsPrev;

    double dir = in_angleError > 0 ? 1 : -1;

    rotF = abs(in_angleError) * dir;

    if (abs(in_angleError) < CREATURE_DELTA_ANGLE_ROT_DEGS) {
        fwdF = 1;
    }

    rotF = (CREATURE_ROT_P_K * abs(in_angleError)) * dir + CREATURE_ROT_D_K * angleErrorDelta;

    deltaDegsPrev = in_angleError;

    angleError = in_angleError;
    targetDistance = VecLen( subVec(in_targetPos, in_pos) );
}

struct Neuron;

struct Link
{
    Link(Neuron* neuron, double weight)
    {
        this->neuron = neuron;
        this->weight = weight;
    }

    Neuron* neuron;
    double weight;
};

struct Neuron
{
    bool signalCalculated = false;
    double output = 0;

    QVector<Link*> links;

    double Signal()
    {
         if (signalCalculated) { return output; }

         double out = 0;

         for (Link* link : links) {
             out += link->neuron->Signal() * link->weight;
         }

         signalCalculated = true;
         output = out;
         return output;

    }

    // for input neurons
    void setSignal(double value) {
        signalCalculated = true;
        output = value;
    }
};

static bool nn_setup_done = false;

Neuron ni_angleError;
Neuron ni_fwdf;

Neuron no_rotF;
Neuron no_fwdf;

QVector<Neuron*> neurons;

/* ni_angleError --> no_rotF
 * ni_fwdf       --> no_fwdf
 */

void Creature::calcWeights()
{

}

void Creature::control_NN(double in_w,
                          double in_orientation,
                          Vec2   in_pos,
                          double in_vel,
                          float  in_rotF,
                          float  in_fwdF,
                          Vec2   in_targetPos,
                          double in_angleError,
                          double in_targetDistance)
{

    if (!nn_setup_done) {
        nn_setup_done = true;

        neurons.append(&ni_angleError);
        neurons.append(&ni_fwdf      );
        neurons.append(&no_rotF      );
        neurons.append(&no_fwdf      );

        no_rotF.links.append(new Link(&ni_angleError, 1.0));
        no_fwdf.links.append(new Link(&ni_fwdf      , 1.0));
    }

    for (Neuron* n : neurons) { n->signalCalculated = false; }

    ni_angleError.setSignal(in_angleError);
    ni_fwdf      .setSignal(1.0);

    rotF = no_rotF.Signal();
    fwdF = no_fwdf.Signal();
}

// ideas:
/*
 * fitness criteria:
 * minimize time to target (TTT):
 *  T   = targetChange-targetChange
 *  D   = targetDistance-targetDistancePrev
 *  TTT = D/T
 *
 * random weights calculation
 * 0.1 step [-1..1]
 *
 * ! NO NN resources freed upon shutdown, DNC for now
 *
 */

#endif // CREATURE_H
