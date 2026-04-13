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
#define NN_W 3

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

//Neuron* nn[NN_H][NN_W] = {{                      0, new Neuron(), new Neuron(), 0                     },
//                          { /* --> */ new Neuron(), new Neuron(), new Neuron(), new Neuron() /* --> */},
//                          {                      0, new Neuron(), new Neuron(), 0                     },};

// --------------------------------------------------------------------------------------------
//                                       in    -->      n    -->     out                     //
Neuron* nn[NN_H][NN_W] = {{                      0,            0,                       },   //
                          { /* --> */ new Neuron(), new Neuron(), new Neuron() /* --> */},   //
                          {                      0,            0,                       },}; //
// --------------------------------------------------------------------------------------------

QVector<Neuron*> neurons;
QVector<Link*> linksGlob;
QVector<Link*> hidLinks;

Neuron* inputN  = nn[1][0];
Neuron* outputN = nn[1][2];

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

                // link is connected to hidden layer
                if (x != 0 && x != (NN_W-1)) { hidLinks.append(lnk); }
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

// training method : gradient descent

// fittness function : input == output * 0.5

// EPOCH : generate weights, perform all experiments, calculate average fitness, report results

#define NUM_EXPERIMENTS     1000 // perform all experiments (values are the dataset)
#define EXPERIMENT_VAL_STEP 0.01  // 0.0 .. 1.0

#define HIDLINK_INIT_MAX 0.1

#define DELTA_WEIGHT_INIT_MIN 0.0
#define DELTA_WEIGHT_INIT_MAX 0.03

#define NOISE_WEIGHT_MIN -0.01
#define NOISE_WEIGHT_MAX  0.01

#define CRITERIA 0.05

bool trainingFinished = false;

// /* no finishing for now */ #define BEST_CRITERIA   0.01 // abs(in-out) average difference per epoch

int epochNum = 0;
int experimentNum = 0;
double avgEpochFitnes = 0.0;

double computeNN(double in)
{
    for (Neuron* n : neurons) { n->signalCalculated = false; }
    inputN->setSignal(in);
    return outputN->Signal();
}

void initNDimVecRandom(QVector<double>* v, double min, double max) { for (int i = 0; i < v->size(); i++) { (*v)[i] = fRand(min, max); } }
void addNDimVecVal    (QVector<double>* v, double val            ) { for (int i = 0; i < v->size(); i++) { (*v)[i] += val;            } }
void mulNDimVecVal    (QVector<double>* v, double val            ) { for (int i = 0; i < v->size(); i++) { (*v)[i] *= val;            } }
void addNDimVecVec    (QVector<double>* a, QVector<double>* b    ) { for (int i = 0; i < a->size(); i++) { (*a)[i] += (*b)[i];        } }

void applyLinkWeightsToVec(QVector<Link*>* lnks, QVector<double>* v) {
    int i = 0; for (Link* l : *lnks) { (*v)[i] = l->weight; i++; }
}

void applyVecToLinkWeights(QVector<double>* v, QVector<Link*>* lnks) {
    int i = 0; for (Link* l : *lnks) { l->weight = (*v)[i]; i++; }
}

int main(int argc, char *argv[])
{
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

    // initialize all link weights to 1
    for (Link* l : linksGlob) { l->weight = 1.0; }
    // initialize hidden layer links' weights to small random
    for (Link* l : hidLinks) { l->weight = fRand(0.0, HIDLINK_INIT_MAX); }

    // === gradient descent vars ===

    QVector<double> weights = QVector<double>(hidLinks.size());
    applyLinkWeightsToVec(&hidLinks, &weights);

    double prevF = 0.0; // prev verageEpochFitness (the function to minimize)
    QVector<double> dw = QVector<double>(hidLinks.size()); // delta weights
    initNDimVecRandom(&dw, DELTA_WEIGHT_INIT_MIN, DELTA_WEIGHT_INIT_MAX);

    bool s1 = false; // last 3 results sign, used to refine the stepping during descent
    bool s2 = false; // last 3 results sign, used to refine the stepping during descent
    bool s3 = false; // last 3 results sign, used to refine the stepping during descent

    while (running) {

        SDL_PumpEvents();
        processInputMainNN();

        //if (!upressed) { continue; }

// ======================================================================================
// ============================= train the NN ===========================================
// ======================================================================================

        // generate weights, perform all experiments, calculate average fitness, report results

        // check epoch finished

        if (!trainingFinished && experimentNum >= NUM_EXPERIMENTS) {
            experimentNum = 0;

            avgEpochFitnes /= NUM_EXPERIMENTS;
            double F = avgEpochFitnes; // the function to minimize
            if (experimentNum == 0) { prevF = F + 0.000001; } // add some epsilon for the firts epoch

            // === === ===
            // update link weights using gradient descent

            double dF = F - prevF;

            if (dF > 0) { mulNDimVecVal(&dw, -1.0); } // change delta direction to opposite if the F has increased

            QVector<double> noise = QVector<double>(hidLinks.size());
            initNDimVecRandom(&noise, NOISE_WEIGHT_MIN, NOISE_WEIGHT_MAX);
            addNDimVecVec(&dw, &noise);

            s1 = s2;
            s2 = s3;
            s3 = dF > 0;

            if (s1 == s3 && s1 != s2) { // we're roaming between some 2 values around the desired minimum
                mulNDimVecVal(&dw, 0.5); // decrease step
            }

            addNDimVecVec(&weights, &dw);
            applyVecToLinkWeights(&weights, &hidLinks);

            // === === ===
            qDebug() << epochNum << "| EPOCH RESULTS" << avgEpochFitnes << hidLinks[0]->weight;

            if (avgEpochFitnes <= CRITERIA) { trainingFinished = true; }

            avgEpochFitnes = 0.0;
            epochNum++;
        }

        if (trainingFinished && experimentNum >= NUM_EXPERIMENTS) { experimentNum = 0; }

        double desiredOutput =           experimentNum / (double)NUM_EXPERIMENTS;
        double      nnOutput = computeNN(experimentNum / (double)NUM_EXPERIMENTS);

        qDebug() << desiredOutput << nnOutput;

        double FTN = abs(nnOutput - desiredOutput);

        avgEpochFitnes += FTN;

        experimentNum++;

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

        std::this_thread::sleep_for(std::chrono::microseconds(simLoopSleepUs));

        frameNum++;
    }

    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
