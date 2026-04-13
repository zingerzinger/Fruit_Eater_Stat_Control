#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string.h>
#include <chrono>
#include <math.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <QDebug>
#include <QVector>

#include "defines.h"
#include "utils.h"

using namespace std;

SDL_Window* window;
SDL_Renderer* renderer;

bool running = true;

int mx, my;

bool upressed = false;
bool dpressed = false;
bool lpressed = false;
bool rpressed = false;

int simLoopSleepUs;
int simSkipFrames;

TTF_Font* font;

void processInputMainNN()
{
    SDL_GetMouseState(&mx, &my);

    SDL_Event event;
    while (SDL_PollEvent(&event)) {

        switch (event.type) {

            case SDL_MOUSEBUTTONDOWN: {

                int mX, mY;
                SDL_GetMouseState(&mX, &mY);

            } break;

            case SDL_KEYDOWN: {
                if (event.key.keysym.sym == SDLK_ESCAPE) { running = false; }

                if (event.key.keysym.sym == SDLK_UP   ) { upressed =  true; }
                if (event.key.keysym.sym == SDLK_DOWN ) { dpressed =  true; }
                if (event.key.keysym.sym == SDLK_LEFT ) { lpressed =  true; }
                if (event.key.keysym.sym == SDLK_RIGHT) { rpressed =  true; }

            } break;

            case SDL_KEYUP: {

                //if (event.key.keysym.sym == SDLK_UP   ) { upressed = false; }
                if (event.key.keysym.sym == SDLK_DOWN ) { dpressed = false; }
                if (event.key.keysym.sym == SDLK_LEFT ) { lpressed = false; }
                if (event.key.keysym.sym == SDLK_RIGHT) { rpressed = false; }

            } break;
        }
    }
}

// === NN ===

#define NN_H 3
#define NN_W 4

#define NN_NEURON_SIDE 20
#define NN_RENDER_SPACING 40
#define NN_RENDER_OFFSET 100

/*           * 100%
 *         * - ACTIVATION_THRESHOLD
 *       * N%
 *     * N%
 *   * - ACTIVATION_THRESHOLD
 * * 0%
 */
#define ACTIVATION_THRESHOLD 0.1

struct Neuron;

struct Link
{
    Link(Neuron* neuronFrom, Neuron* neuronTo, double weight)
    {
        this->neurFrom = neuronFrom;
        this->neurTo   = neuronTo;
        this->weight = weight;
    }

    Neuron* neurFrom;
    Neuron* neurTo;
    double weight;
};

struct Neuron
{
    bool signalCalculated = false;
    double output = 0;

    int xRendCoord;
    int yRendCoord;

    int xid;
    int yid;

    bool isInput;
    bool isOutput;

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
             out += link->neurFrom->Signal() * link->weight;
         }

         if (out <       ACTIVATION_THRESHOLD) { out = 0.0; }
         if (out > 1.0 - ACTIVATION_THRESHOLD) { out = 1.0; }

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

Neuron* nn[NN_H][NN_W] = {{                      0, new Neuron(), new Neuron(), 0                     },
                          { /* --> */ new Neuron(), new Neuron(), new Neuron(), new Neuron() /* --> */},
                          {                      0, new Neuron(), new Neuron(), 0                     },};

QVector<Neuron*> neurons;
QVector<Link*> linksGlob;

Neuron* inputN  = nn[1][0];
Neuron* outputN = nn[1][3];

// === === ===

void initNNLinks()
{
    for (int y = 0; y < NN_H; y++) {
        for (int x = 0; x < NN_W; x++) {

            Neuron* nCur = nn[y][x];
            if (!nCur) { continue; }

            neurons.append(nCur);

            nCur->xid = x;
            nCur->yid = y;

            nCur->xRendCoord = x * NN_RENDER_SPACING + NN_RENDER_OFFSET;
            nCur->yRendCoord = y * NN_RENDER_SPACING + NN_RENDER_OFFSET;

            for (int line = 0; line < NN_H; line++) {

                if (x == 0) { continue; }
                Neuron* nPrev = nn[line][x-1];
                if (!nPrev) { continue; }

                Link* lnk = new Link(nPrev, nCur, 0.0);
                nCur->links.append(lnk);
                linksGlob.append(lnk);
            }
        }
    }
}

void renderNN(Neuron* nn[NN_H][NN_W], int offset)
{
    SDL_Rect neuronVisRect;
    neuronVisRect.w = NN_NEURON_SIDE;
    neuronVisRect.h = NN_NEURON_SIDE;

    // render neurons

    for (int y = 0; y < NN_H; y++) {
        for (int x = 0; x < NN_W; x++) {

            Neuron* n = nn[y][x];
            if (!n) { continue; }

            SDL_SetRenderDrawColor(renderer, 0, n->output * 255.0, 0, 0);
            neuronVisRect.x = x * NN_RENDER_SPACING - (NN_NEURON_SIDE / 2) + offset;
            neuronVisRect.y = y * NN_RENDER_SPACING - (NN_NEURON_SIDE / 2) + offset;
            SDL_RenderFillRect(renderer, &neuronVisRect);
        }
    }

    // render links

    for (int y = 0; y < NN_H; y++) {
        for (int x = 0; x < NN_W; x++) {

            Neuron* n = nn[y][x];
            if (!n) { continue; }

            for (Link* l : n->links) {

                SDL_SetRenderDrawColor(renderer, 0, 0, l->weight * 255.0, 0);

                SDL_RenderDrawLine(renderer, l->neurFrom->xRendCoord, l->neurFrom->yRendCoord,
                                             l->neurTo  ->xRendCoord, l->neurTo  ->yRendCoord);
            }
        }
    }
}

// === === ===

// training method : random weights

// fittness function : input == output * 0.5

// 0..NUM_EPOCHS : generate weights, perform all experiments, calculate average fitness, report results

#define NUM_EPOCHS          1000  // generate weights
#define NUM_EXPERIMENTS     10  // perform all experiments (values are the dataset)
#define EXPERIMENT_VAL_STEP 0.01  // 0.0 .. 1.0

#define BEST_CRITERIA   0.02 // abs(in-out) average difference per epoch

#define BST_SHOW_US 250000

int epochNum = 0;
int experimentNum = 0;
double avgEpochFitnes = 0.0;
double minFTN = MAXFLOAT;
int numBest = 0;
bool showNewBest = false;
bool trainingFinished = false;

QVector<Neuron*> trainedNeurons;
QVector<Link*> trainedLinks;

Neuron* trainedInputN  = nullptr;
Neuron* trainedOutputN = nullptr;

void experiment(double in, double out)
{
    for (Neuron* n : neurons) { n->signalCalculated = false; }

    inputN->setSignal(in);
    double NNval = outputN->Signal();

    double FTN = abs(NNval - out);
    avgEpochFitnes += FTN;

    if (FTN <= BEST_CRITERIA) {
        numBest++;
    }
}

double calcOut(double in)
{
    for (Neuron* n : trainedNeurons) { n->signalCalculated = false; }
    trainedInputN->setSignal(in);
    return trainedOutputN->Signal();
}

int main(int argc, char *argv[])
{
    // === gradient minimum search ===

    // f = x^2 + y^2

    double f;
    double fp;

    Vec2 v(-1000, 100);
    Vec2 dv(fRand(-0.3, 0.3), fRand(-0.3, 0.3)); // dir
    Vec2 tv(0, 0);

    double dfPrev = 1;

    bool s1 = false;
    bool s2 = false;
    bool s3 = false;

    int steps = 0;

    while (1) {

        steps++;

        fp = v.x*v.x + v.y*v.y;
        tv = addVec(v, dv);
        f  = tv.x*tv.x + tv.y*tv.y;

        double df = f - fp;

        qDebug() << steps << "|" << f << "|" << df << "|" << (int)abs(df / dfPrev * 100.0) << "|" << ( QString("{%1,%2}").arg(v.x).arg(v.y) );
        dfPrev = df;

        dv = (df > 0 /*growth*/) ? mulVecScalar(-1.0, dv) : dv;
        dv = addVec(dv, Vec2(fRand(-0.01, 0.01), fRand(-0.01, 0.01))); // divert from straight line a bit, might use the crossProduct actually (?)

        s1 = s2;
        s2 = s3;
        s3 = df > 0;

        if (s1 == s3 && s1 != s2) { // we're roaming between some 2 values
            dv = mulVecScalar(0.5, dv); // decrease step
        }

        v = addVec(v, dv);

        if (abs(f) < 0.01) { return 0; }

        //std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    // === === ===

    // y = x^2
    // start : x = -10
    // find y global minimum

    double xprev;
    double x = 10.0;
    double step = 0.01;
    double yprev;
    double y;

    double criteria = 0.01;

    double prevDeltaY = 0;

    y = x*x;
    yprev = y;

    while (true)
    {
        yprev = x*x;
        x+= step;
        y = x*x;

        if (y > yprev) {
            step = -step;
        }

        double deltaY = abs(y-yprev);
        if (deltaY > prevDeltaY) { step *= 0.5; }
        prevDeltaY = deltaY;

        qDebug() << step << "|"<< x << "|" << y;

        if (deltaY <= criteria) { break; }
    }

    return 0;
    // === === ===

    simLoopSleepUs = 50;// 1ms //LOOP_SLEEP_US;
    simSkipFrames  = SIM_SKIP_RENDER_FRAMES;

    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1" );

    window = SDL_CreateWindow("main", 650, 450, W_W, W_H, SDL_WINDOW_BORDERLESS | SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, 0);

    TTF_Init();
    font = TTF_OpenFont("Monoid-Bold.ttf", 12);
    if (font == NULL) { fprintf(stderr, "error: font not found\n"); exit(EXIT_FAILURE); }

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    uint64_t frameNum = 0;

    initNNLinks();

    inputN ->isInput  = true;
    outputN->isOutput = true;

    // generate random weights
    for (Link* l : linksGlob) { l->weight = fRand(0.0, 1.0); }

    while (running) {

        SDL_PumpEvents();
        processInputMainNN();

        //if (!upressed) { continue; }

// ======================================================================================
// ============================= train the NN ===========================================
// ======================================================================================

        if (trainingFinished) { goto lblTrainingFinished; }

        // 0..NUM_EPOCHS : generate weights, perform all experiments, calculate average fitness, report results

        if (experimentNum > NUM_EXPERIMENTS) { // next epoch

            avgEpochFitnes /= NUM_EXPERIMENTS;

            if (/*avgEpochFitnes <= BEST_CRITERIA && avgEpochFitnes < minFTN*/ numBest >= (int)(NUM_EXPERIMENTS * (1.0 - BEST_CRITERIA))) {
                minFTN = avgEpochFitnes;
                numBest = 0;

                qDeleteAll(trainedNeurons); trainedNeurons.clear();
                qDeleteAll(trainedLinks  ); trainedLinks  .clear();
                for (Neuron* n : neurons) { trainedNeurons.append(new Neuron(*n)); }
                for (Link* l : linksGlob) { trainedLinks  .append(new Link  (*l)); }

                // neuron --> links, io

                for (Neuron* n : trainedNeurons) {
                    if (n->isInput ) { trainedInputN  = n; }
                    if (n->isOutput) { trainedOutputN = n; }
                }

                for (Link* l : trainedLinks) {

                    for (Neuron* n : trainedNeurons) {

                        if (l->neurFrom->xid == n->xid &&
                            l->neurFrom->yid == n->yid) {
                            l->neurFrom = n;
                        }

                        if (l->neurTo  ->xid == n->xid &&
                            l->neurTo  ->yid == n->yid) {
                            l->neurTo   = n;
                        }

                    }
                }

                showNewBest = true;
            }

            qDebug() << "EPCH" << epochNum << "FTN" << avgEpochFitnes << "BST" << minFTN;

            avgEpochFitnes = 0.0;
            experimentNum = 0;
            epochNum++;
            // generate random weights
            for (Link* l : linksGlob) { l->weight = fRand(0.0, 1.0); }
        }

        if (epochNum > NUM_EPOCHS) {
            // --> file
            trainingFinished = true;
            goto lblTrainingFinished;
        }

        experiment(experimentNum / (double)NUM_EXPERIMENTS, experimentNum / (double)NUM_EXPERIMENTS * 0.5);

        experimentNum++;

        lblTrainingFinished:

// ======================================================================================
// ======================================================================================
// ======================================================================================

        SDL_SetRenderDrawColor(renderer, 64, 64, 64, 0);
        SDL_RenderClear(renderer);

        renderNN(nn, NN_RENDER_OFFSET);

        // render screen border
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 0);
        SDL_Rect wborderRect;
        wborderRect.x = 0;
        wborderRect.y = 0;
        wborderRect.w = W_W-1;
        wborderRect.h = W_H-1;
        SDL_RenderDrawRect(renderer, &wborderRect);

        SDL_RenderPresent(renderer);

        if (showNewBest) {
            showNewBest = false;
            //std::this_thread::sleep_for(std::chrono::microseconds(BST_SHOW_US));
            qDebug() << "NEW BEST";
            qDebug() << "==========================================================";
            qDebug() << "==========================================================";

            // save the best solution

            for (Neuron* n : neurons) {
                QString io = n->isInput ? "I" : (n->isOutput ? "O" : "");
                qDebug() << "ID" << n->yid << n->xid << "IO" << io;
            }

            for (Link* l : linksGlob) {
                QString s = QString("([%1][%2]) --> ([%3][%4]) | W %5").arg(l->neurFrom->yid)
                                                                      .arg(l->neurFrom->xid)
                                                                      .arg(l->neurTo  ->yid)
                                                                      .arg(l->neurTo  ->xid)
                                                                      .arg(l->weight);

                qDebug() << s;
            }

            qDebug() << "==========================================================";
            qDebug() << "==========================================================";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(simLoopSleepUs));

        if (trainingFinished) {
            double inVal  = fRand(0, 1);
            double outVal = calcOut(inVal);
            qDebug() << inVal << outVal << (abs(inVal - outVal));
        }

        frameNum++;
    }

    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
