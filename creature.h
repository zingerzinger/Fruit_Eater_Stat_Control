#ifndef CREATURE_H
#define CREATURE_H

#include <QQueue>

#include "defines.h"
#include "utils.h"

// === NN ===

// ideas:
/*
 * fitness criteria:
 * minimize time to target (TTT) (maximize speed):
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

    void normalizeWeights()
    {
        double weights = 0.0;

        for (Link* link : links) {
            weights += link->weight;
        }

        for (Link* link : links) {
            link->weight /= weights;
        }
    }

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

    void control_NN(double w,
                    double orientation,
                    Vec2   pos,
                    double vel,
                    float  rotF,
                    float  fwdF,
                    Vec2   targetPos,
                    double angleError,
                    double angleErrorDelta,
                    double targetDistance);

    // === === ===

    Neuron ni_angleError;
    Neuron ni_angleDError;
    Neuron ni_fwdf;

    Neuron no_rotF;
    Neuron no_fwdf;

    QVector<Neuron*> neurons;

    /* ni_angleError --> no_rotF
     * ni_fwdf       --> no_fwdf
     */

    double maxTTT = 0.0;
    double bestWeight = 0.0;

    int numEpoch = 0;
    double cTTT = 0;
    double cweight = 0.01;
    double ccweight = 1.0;

    bool IpassOk  = false;
    bool IIpassOk = false;

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

#define EPOCHS_PER_WEIGHT 2

void Creature::calcWeights()
{
    int64_t T = frameNum - frameNumPrev;
    double  D = targetDistancePrev;
    double  TTT = D / T;

    cTTT += TTT;

    frameNumPrev = frameNum;
    targetDistancePrev = targetDistance;

    // === === ===

    qDebug() << no_rotF.links[0]->weight << "|" << no_rotF.links[1]->weight << "|" << TTT;

    numEpoch++;

    if (!IpassOk) {

        no_rotF.links[0]->weight = cweight;
        //no_rotF.normalizeWeights();

        if (numEpoch % EPOCHS_PER_WEIGHT == 0) {
            qDebug() << "ITER";
            cTTT = 0;
            cweight += 0.1;
            if (cweight >= 1.0) { cweight = 0.0; IpassOk = true; }
        }
    }

    if (IpassOk && !IIpassOk) {

        no_rotF.links[0]->weight = ccweight;
        no_rotF.links[1]->weight = cweight;
        //no_rotF.normalizeWeights();

        if (numEpoch % EPOCHS_PER_WEIGHT == 0) {
            qDebug() << "ITER";
            cTTT = 0;
            cweight += 5.0;
            ccweight -= 0.025;
            if (cweight >= 50.0) { cweight = 1.0; IIpassOk = true; }
        }
    }
}

void Creature::control_NN(double in_w,
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
    for (Neuron* n : neurons) { n->signalCalculated = false; }

    ni_angleError .setSignal(in_angleError);
    ni_angleDError.setSignal(angleErrorDelta);
    ni_fwdf       .setSignal(1.0);

    rotF = no_rotF.Signal();
    fwdF = no_fwdf.Signal();
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

    if (!nn_setup_done) {
        nn_setup_done = true;

        neurons.append(&no_rotF      );
        neurons.append(&no_fwdf      );

        no_rotF.links.append(new Link(&ni_angleError , 0.0));
        no_rotF.links.append(new Link(&ni_angleDError, 0.0));
        no_fwdf.links.append(new Link(&ni_fwdf       , 1.0));
    }

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
        calcWeights();
        frameNumPrev = frameNum;
        targetDistancePrev = targetDistance;
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

    rotF = (CREATURE_ROT_P_K * abs(in_angleError)) * dir - CREATURE_ROT_D_K * angleErrorDelta;

//    deltaDegsPrev = in_angleError;
//
//    angleError = in_angleError;
}

#endif // CREATURE_H
