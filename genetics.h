#ifndef GENETICS_H
#define GENETICS_H

#include <QDebug>
#include <QVector>

#include "defines.h"
#include "sim.h"
#include "creature.h"

#define GEN_PARAM_MIN -100.0
#define GEN_PARAM_MAX  100.0

#define GEN_PARAM_MUTATION_MIN -10.0
#define GEN_PARAM_MUTATION_MAX  10.0

#define GEN_STORE_MAX_GENERATIONS 32

//#define GEN_TIME_MAX_S (5 * 60) // 5 minutes

typedef struct
{
    double PS;
    double DS;
    double IS;
    double PT;
    double TTD; // time to death
} GCODE;

class Genetics
{
public:
    Genetics(Sim* sim, Creature* creature);
    ~Genetics();

    void step(double dt_secs);

    QVector<GCODE> generations;

    Sim* sim;
    Creature* creature;

    GCODE cgcode;

    double stime = 0.0;
    double ctime = 0.0;
};

Genetics::Genetics(Sim* sim, Creature* creature)
{
    this->sim = sim;
    this->creature = creature;

    creature->ROT_P_K = fRand(GEN_PARAM_MIN, GEN_PARAM_MAX);
    creature->ROT_D_K = fRand(GEN_PARAM_MIN, GEN_PARAM_MAX);
    creature->ROT_I_K = fRand(GEN_PARAM_MIN, GEN_PARAM_MAX);
    creature->SPD_P_K = fRand(            0, GEN_PARAM_MAX);

    creature->food = 0.0;
}

Genetics::~Genetics()
{

}

void Genetics::step(double dt_secs)
{
    ctime += dt_secs;

    if (creature->food <= 0.0) {

        // record new GCODE
        cgcode.PS = creature->ROT_P_K;
        cgcode.DS = creature->ROT_D_K;
        cgcode.IS = creature->ROT_I_K;
        cgcode.PT = creature->SPD_P_K;
        cgcode.TTD = ctime - stime;
        stime = ctime;

        generations.append(cgcode);

        sim->reset();
        creature->reset();
        sim->addFruitsRandom();

        if (generations.size() < 2) {

            creature->ROT_P_K = fRand(GEN_PARAM_MIN, GEN_PARAM_MAX);
            creature->ROT_D_K = fRand(GEN_PARAM_MIN, GEN_PARAM_MAX);
            creature->ROT_I_K = fRand(GEN_PARAM_MIN, GEN_PARAM_MAX);
            creature->SPD_P_K = fRand(            0, GEN_PARAM_MAX);

        } else {

            // sort descending by time to death
            std::sort(generations.begin(), generations.end(), [](const GCODE &a, const GCODE &b) {
                return a.TTD >= b.TTD;
            });

            // mix best and other random

            int bestIdx  = 0;
            int otherIdx = 1;

            creature->ROT_P_K = generations[bestIdx].PS * 0.5 + generations[otherIdx].PS * 0.5 + fRand(GEN_PARAM_MUTATION_MIN, GEN_PARAM_MUTATION_MAX);
            creature->ROT_D_K = generations[bestIdx].DS * 0.5 + generations[otherIdx].DS * 0.5 + fRand(GEN_PARAM_MUTATION_MIN, GEN_PARAM_MUTATION_MAX);
            creature->ROT_I_K = generations[bestIdx].IS * 0.5 + generations[otherIdx].IS * 0.5 + fRand(GEN_PARAM_MUTATION_MIN, GEN_PARAM_MUTATION_MAX);
            creature->SPD_P_K = generations[bestIdx].PT * 0.5 + generations[otherIdx].PT * 0.5 + fRand(GEN_PARAM_MUTATION_MIN, GEN_PARAM_MUTATION_MAX);
        }

        if (generations.size() >= GEN_STORE_MAX_GENERATIONS) { generations.removeLast(); }

        qDebug() << "=============================";
        for (GCODE gc : generations)  {

            qDebug() << ( QString("%1|%2|%3|%4|%5").arg(gc.TTD, 0, 'f', 3)
                                                   .arg(gc.PS , 0, 'f', 3)
                                                   .arg(gc.IS , 0, 'f', 3)
                                                   .arg(gc.DS , 0, 'f', 3)
                                                   .arg(gc.PT , 0, 'f', 3) );
        }
    }
}

#endif // GENETICS_H
