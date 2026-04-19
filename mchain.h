#ifndef MCHAIN_H
#define MCHAIN_H

#include <algorithm>

#include <QDebug>
#include <QVector>

#include "defines.h"
#include "utils.h"

class Mchain
{
public:

    Mchain(int numStates);
    ~Mchain();

    void step(double dt_secs);

    void initStateTransitionsEqual();
    void initStateTransitionsRandom();

    void chooseMostProbableState();

    void increaseProbabilityByAmount(int pstate, int nstate, double amount /* 0.0 .. 1.0 */);

    void print();

    //QVector<int> states;

    // state idx --> state idx
    QVector<QVector<double>> transitionProbabilities;

    int numStates;

    // hmm, DNC about the access modifiers, I am alone here anyway
    int cstate = 0;

    static const int MAX_HISTORY_ITEMS = 256;

    QVector<int> stateHistory;

};

Mchain::Mchain(int numStates)
{
    //states.resize(numStates);

    this->numStates = numStates;

    for (int y = 0; y < numStates; y++) {
        QVector<double> row;
        for (int x = 0; x < numStates; x++) { row.append(0.0); }
        transitionProbabilities.append(row);
    }

    initStateTransitionsEqual();
    //initStateTransitionsRandom();

    print();
}

Mchain::~Mchain()
{

}

void Mchain::print()
{
    qDebug() << "=========================================";
    for (int y = 0; y < numStates; y++) {
        QString s;
        for (int x = 0; x < numStates; x++) {
            s += QString("%1 |").arg(transitionProbabilities[y][x], 0, 'f', 3);
        }
        qDebug() << s;
    }
    qDebug() << "=========================================";
}

void Mchain::initStateTransitionsEqual()
{
    for (int y = 0; y < numStates; y++) {
        for (int x = 0; x < numStates; x++) {
            transitionProbabilities[y][x] = 1.0 / numStates;
        }
    }
}

void Mchain::initStateTransitionsRandom()
{
    for (int y = 0; y < numStates; y++) {
        double total = 1.0;
        for (int x = 0; x < numStates; x++) {
            double cprobability = fRand(0.0, total);
            if (x == numStates - 1) { cprobability = total; }
            total -= cprobability;

            transitionProbabilities[y][x] = cprobability;
        }
    }
}

void Mchain::increaseProbabilityByAmount(int pstate, int nstate, double amount /* 0.0 .. 1.0 */)
{
    double curProbability = transitionProbabilities[pstate][nstate];
    double increase = amount;

    if (curProbability + amount >= 1.0) {
        increase = 1.0 - curProbability;
        curProbability = 1.0;
    } else {
        curProbability += increase;
    }

    // update pstate --> nstate transition probability
    transitionProbabilities[pstate][nstate] = curProbability;

    // distribute increase among other probabilities
    double decrease = increase / (numStates-1);

    for (int i = 0; i < numStates; i++) {
        if (i != nstate) { transitionProbabilities[pstate][i] -= decrease; }
    }
}

void Mchain::chooseMostProbableState()
{
    QVector<double> nextStateProbabilities = transitionProbabilities[cstate];

    // ascending
    std::sort(nextStateProbabilities.begin(), nextStateProbabilities.end());

    QVector<double> segments;
    double accumulated = 0.0;
    for (double p : nextStateProbabilities) {
        accumulated += p;
        segments.append(accumulated);
    }

    int nextState = 0;
    double probability = fRand(0.0, 1.0);
    for (double p : segments) {
        if (probability < p) { break; }
        nextState++;
    }

    cstate = nextState;
}

void Mchain::step(double dt_secs)
{
    chooseMostProbableState();
}

#endif // MCHAIN_H
