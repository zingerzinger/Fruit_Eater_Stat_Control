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

//         if (out <       ACTIVATION_THRESHOLD) { out = 0.0; }
//         if (out > 1.0 - ACTIVATION_THRESHOLD) { out = 1.0; }

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

#define NN_H 3
#define NN_W 4

// ---------------------------------------------------------------------------------------------------------
//                                       in    -->      n    -->      n    -->     out                    //
Neuron* nn[NN_H][NN_W] = {{                      0, new Neuron(), new Neuron(), 0                     },  //
                          { /* --> */ new Neuron(), new Neuron(), new Neuron(), new Neuron() /* --> */},  //
                          {                      0, new Neuron(), new Neuron(), 0                     },};//
// -------------------------------------------------------------------------------------------------------//

//// --------------------------------------------------------------------------------------------
////                                       in    -->      n    -->     out                     //
//Neuron* nn[NN_H][NN_W] = {{                      0, new Neuron(),                       },   //
//                          { /* --> */ new Neuron(), new Neuron(), new Neuron() /* --> */},   //
//                          {                      0, new Neuron(),                       },}; //
//// --------------------------------------------------------------------------------------------

QVector<Neuron*> neurons;
QVector<Link*> linksGlob;
QVector<Link*> hidLinks;

QVector<QVector<Link*>> hidLinkLayers; // input lyer <-- output layer

Neuron* inputN  = nn[1][   0  ];
Neuron* outputN = nn[1][NN_W-1];

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

    for (int x = 0; x < NN_W; x++) {

        QVector<Link*> layerLinks;

        for (int y = 0; y < NN_H; y++) {

            if (x == 0) { continue; }
            Neuron* n = nn[y][x];
            if (!n) { continue; }

            layerLinks.append(n->links);
        }

        if (!layerLinks.empty()) { hidLinkLayers.append(layerLinks); }
        layerLinks.clear();
    }

    std::reverse(hidLinkLayers.begin(), hidLinkLayers.end());
}

void renderNN(Neuron* nn[NN_H][NN_W], int offset)
{
    SDL_Rect neuronVisRect;
    neuronVisRect.w = NN_NEURON_SIDE;
    neuronVisRect.h = NN_NEURON_SIDE;

    // render neurons

    // get min and max neuron values

    double min =  MAXFLOAT;
    double max = -MAXFLOAT;

    for (Neuron* n : neurons) {
        if (n == inputN || n == outputN) { continue; }
        if (n->output < min) { min = n->output; }
        if (n->output > max) { max = n->output; }
    }

    double range = max - min;

    for (int y = 0; y < NN_H; y++) {
        for (int x = 0; x < NN_W; x++) {

            Neuron* n = nn[y][x];
            if (!n) { continue; }

            int c = (n->output - min) / range;

            if      (n == inputN ) { SDL_SetRenderDrawColor(renderer, 0, n->output * 255, 0, 0); }
            else if (n == outputN) { SDL_SetRenderDrawColor(renderer, 0, n->output * 255, 0, 0); }
            else                   { SDL_SetRenderDrawColor(renderer, 0,         c * 255, 0, 0); }

            neuronVisRect.x = x * NN_RENDER_SPACING - (NN_NEURON_SIDE / 2) + offset;
            neuronVisRect.y = y * NN_RENDER_SPACING - (NN_NEURON_SIDE / 2) + offset;
            SDL_RenderFillRect(renderer, &neuronVisRect);
        }
    }

    // render links

    min =  MAXFLOAT;
    max = -MAXFLOAT;

    // gather hidden links weights
    for (int y = 0; y < NN_H; y++) {
        for (int x = 1; x < NN_W-1; x++) {

            Neuron* n = nn[y][x];
            if (!n) { continue; }

            for (Link* l : n->links) {
                if (l->weight < min) { min = l->weight; }
                if (l->weight > max) { max = l->weight; }
            }
        }
    }

    range = max - min;

    for (int y = 0; y < NN_H; y++) {
        for (int x = 0; x < NN_W; x++) {

            Neuron* n = nn[y][x];
            if (!n) { continue; }

            for (Link* l : n->links) {

                int c = (l->weight - min) / range;

                SDL_SetRenderDrawColor(renderer, 0, 0, c * 255, 0);

                SDL_RenderDrawLine(renderer, l->neurFrom->xRendCoord, l->neurFrom->yRendCoord + 0,
                                             l->neurTo  ->xRendCoord, l->neurTo  ->yRendCoord + 0);
                SDL_RenderDrawLine(renderer, l->neurFrom->xRendCoord, l->neurFrom->yRendCoord + 1,
                                             l->neurTo  ->xRendCoord, l->neurTo  ->yRendCoord + 1);
                SDL_RenderDrawLine(renderer, l->neurFrom->xRendCoord, l->neurFrom->yRendCoord + 2,
                                             l->neurTo  ->xRendCoord, l->neurTo  ->yRendCoord + 2);
            }
        }
    }
}

void renderFlag(int r, int g, int b, int x, int y)
{
    SDL_Rect rect;
    rect.w = NN_NEURON_SIDE;
    rect.h = NN_NEURON_SIDE;

    SDL_SetRenderDrawColor(renderer, r, g, b, 0);
    rect.x = x;
    rect.y = y;
    SDL_RenderFillRect(renderer, &rect);
}

// === === ===

// training method : gradient descent

// fittness function : input == output * 0.5

// EPOCH : generate weights, perform all experiments, calculate average fitness, report results

#define NUM_EXPERIMENTS     1000 // perform all experiments (values are the dataset)
#define EXPERIMENT_VAL_STEP 0.01  // 0.0 .. 1.0

#define HIDLINK_INIT_MIN -0.3
#define HIDLINK_INIT_MAX  0.3

#define DELTA_WEIGHT_INIT_MIN -0.01
#define DELTA_WEIGHT_INIT_MAX  0.01

#define NOISE_WEIGHT_MIN -0.005
#define NOISE_WEIGHT_MAX  0.005

#define CRITERIA 0.025

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
    simLoopSleepUs = 16666;// 1ms //LOOP_SLEEP_US;
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
    for (Link* l : linksGlob) { l->weight = fRand(HIDLINK_INIT_MAX, HIDLINK_INIT_MAX); }
    // initialize hidden layer links' weights to small random
//    for (Link* l : hidLinks) { l->weight = fRand(HIDLINK_INIT_MAX, HIDLINK_INIT_MAX); }

    // === gradient descent vars ===

    int layerIdx = 0;

    QVector<double> weights = QVector<double>(hidLinkLayers[layerIdx].size());
    applyLinkWeightsToVec(&(hidLinkLayers[layerIdx]), &weights);

    double prevF = 0.0; // prev verageEpochFitness (the function to minimize)
    QVector<double> dw = QVector<double>(hidLinkLayers[layerIdx].size()); // delta weights
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
            if (experimentNum == 0) { prevF = F + 0.000001; } // add some epsilon for the first epoch

            // === === ===
            // update link weights using gradient descent

            double dF = F - prevF;

            dw = QVector<double>(hidLinkLayers[layerIdx].size()); // delta weights

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

            // update weight vector according to layer links size
            weights = QVector<double>(hidLinkLayers[layerIdx].size());
            applyLinkWeightsToVec(&(hidLinkLayers[layerIdx]), &weights);

            addNDimVecVec(&weights, &dw);
            applyVecToLinkWeights(&weights, &(hidLinkLayers[layerIdx]));

            // === === ===
            qDebug() << epochNum << "| EPOCH RESULTS" << avgEpochFitnes;// << hidLinks[0]->weight;

            if (avgEpochFitnes <= CRITERIA) { trainingFinished = true; }

            layerIdx++; if (layerIdx >= hidLinkLayers.size()) { layerIdx = 0; } // loop layers training

            // update dw for next layer
            //QVector<double> dw = QVector<double>(hidLinkLayers[layerIdx].size()); // delta weights
            //initNDimVecRandom(&dw, DELTA_WEIGHT_INIT_MIN, DELTA_WEIGHT_INIT_MAX);

            avgEpochFitnes = 0.0;
            epochNum++;
        }

        if (trainingFinished && experimentNum >= NUM_EXPERIMENTS) { experimentNum = 0; }

        double desiredOutput =           experimentNum / (double)NUM_EXPERIMENTS;
        double      nnOutput = computeNN(experimentNum / (double)NUM_EXPERIMENTS);

        if (trainingFinished) { qDebug() << desiredOutput << nnOutput; }

        double FTN = abs(nnOutput - desiredOutput);

        avgEpochFitnes += FTN;

        experimentNum++;

// ======================================================================================
// ======================================================================================
// ======================================================================================

        SDL_SetRenderDrawColor(renderer, 64, 64, 64, 0);
        SDL_RenderClear(renderer);

        renderNN(nn, NN_RENDER_OFFSET);

        // render flag training finished
        renderFlag(0, trainingFinished ? 255 : 0, 0, 20, 60);

        // render screen border
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 0);
        SDL_Rect wborderRect;
        wborderRect.x = 0;
        wborderRect.y = 0;
        wborderRect.w = W_W-1;
        wborderRect.h = W_H-1;
        SDL_RenderDrawRect(renderer, &wborderRect);

        //if (trainingFinished) { SDL_RenderPresent(renderer); }
        SDL_RenderPresent(renderer);

        if (!trainingFinished) { std::this_thread::sleep_for(std::chrono::microseconds(10)); }
        if (trainingFinished) { std::this_thread::sleep_for(std::chrono::microseconds(simLoopSleepUs)); }

        //if (trainingFinished) { std::this_thread::sleep_for(std::chrono::microseconds(simLoopSleepUs)); }

        frameNum++;
    }

    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
