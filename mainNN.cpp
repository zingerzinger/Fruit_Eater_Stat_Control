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

                if (event.key.keysym.sym == SDLK_UP   ) { upressed = false; }
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
#define NUM_EXPERIMENTS     250   // perform all experiments (values are the dataset)
#define EXPERIMENT_VAL_STEP 0.01  // 0.0 .. 1.0

#define BEST_CRITERIA   0.025 // abs(in-out) average difference per epoch

#define BST_SHOW_US 250000

int epochNum = 0;
int experimentNum = 0;

double avgEpochFitnes = 0.0;

double minFTN = MAXFLOAT;

bool showNewBest = false;

void experiment(double in, double out)
{
    for (Neuron* n : neurons) { n->signalCalculated = false; }

    inputN->setSignal(in);
    double NNval = outputN->Signal();

    double FTN = abs(NNval - out);
    avgEpochFitnes += FTN;
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

    // generate random weights
    for (Link* l : linksGlob) { l->weight = fRand(0.0, 1.0); }

    while (running) {

        SDL_PumpEvents();
        processInputMainNN();

// ======================================================================================
// ============================= train the NN ===========================================
// ======================================================================================

        // 0..NUM_EPOCHS : generate weights, perform all experiments, calculate average fitness, report results

        if (experimentNum > NUM_EXPERIMENTS) { // next epoch

            avgEpochFitnes /= NUM_EXPERIMENTS;

            if (avgEpochFitnes < minFTN) { minFTN = avgEpochFitnes; }
            if (avgEpochFitnes <= BEST_CRITERIA) { showNewBest = true; }
            qDebug() << "EPCH" << epochNum << "FTN" << avgEpochFitnes << "BST" << minFTN;

            avgEpochFitnes = 0.0;
            experimentNum = 0;
            epochNum++;
            // generate random weights
            for (Link* l : linksGlob) { l->weight = fRand(0.0, 1.0); }
        }

        if (epochNum > NUM_EPOCHS) { goto lblTrainingFinished; }

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
            std::this_thread::sleep_for(std::chrono::microseconds(BST_SHOW_US));
            qDebug() << "NEW BEST";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(simLoopSleepUs));

        frameNum++;
    }

    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
