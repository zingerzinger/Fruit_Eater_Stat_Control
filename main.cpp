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

struct Neuron;

struct Link
{
    Link(Neuron* neuronFrom, Neuron* neuronTo)
    {
        this->neurFrom = neuronFrom;
        this->neurTo   = neuronTo;
    }

    Neuron* neurFrom;
    Neuron* neurTo;

    void print();
};

#define TAN_ONE (1 / 1.5575)

struct Neuron
{
    bool outputCalculated = false;
    double output = 0;

    double weight;

    int xRendCoord;
    int yRendCoord;

    int xid;
    int yid;

    bool isInput;
    bool isOutput;

    QVector<Link*> linksIn;
    QVector<Link*> linksOut;

    double getOutputForward()
    {
         if (outputCalculated) { return output; }

         double out = 0;

         for (Link* link : linksIn) {
             out += link->neurFrom->getOutputForward();
         }

         outputCalculated = true;
         output = activate(out);
         return output;

    }

    double getOutputBackward()
    {
         if (outputCalculated) { return output; }

         double out = 0;

         for (Link* link : linksOut) {
             out += link->neurFrom->getOutputBackward();
         }

         outputCalculated = true;
         output = activate(out);
         return output;
    }

    double activate(double val) {
        // sigmoid : 1 / (1 + e^-x)
        return 1.0 / (1.0 + pow(M_E, -val)) * weight;
    }

    double propagateError() {
        //
    }

    // for input neurons
    void setOutput(double value) {
        outputCalculated = true;
        output = value;
    }

    void print() {
        qDebug() << ( QString("[%1][%2] | W %3").arg(yid)
                                                .arg(xid)
                                                .arg(weight));
    }

    void printLinksIn () { for (Link* l : linksIn ) { l->print(); } }
    void printLinksOut() { for (Link* l : linksOut) { l->print(); } }
};

void Link::print() {
    qDebug() << ( QString("[%1][%2] --> [%3][%4]").arg(neurFrom->yid)
                                                  .arg(neurFrom->xid)
                                                  .arg(neurTo  ->yid)
                                                  .arg(neurTo  ->xid));
}

#define NN_H 1
#define NN_W 3

// ------------------------------------------------------------------------------------------//
//                                       in    -->      n    --     out                      //
Neuron* nn[NN_H][NN_W] = {{ /* --> */ new Neuron(), new Neuron(), new Neuron() /* --> */},}; //
// ------------------------------------------------------------------------------------------//

//// --------------------------------------------------------------------------------------------
////                                       in    -->      n    -->     out                     //
//Neuron* nn[NN_H][NN_W] = {{                      0, new Neuron(),                       },   //
//                          { /* --> */ new Neuron(), new Neuron(), new Neuron() /* --> */},   //
//                          {                      0, new Neuron(),                       },}; //
//// --------------------------------------------------------------------------------------------

QVector<QVector<Neuron*>> layersFwd;
QVector<QVector<Neuron*>> layersBck;

QVector<Neuron*> neurons;
QVector<Link*> links;
QVector<Link*> hidLinks;

Neuron* inputN  = nn[0][   0  ];
Neuron* outputN = nn[0][NN_W-1];

void printLayer(int id)
{
    for (Neuron* n : layersFwd[id]) { n->print(); }
}

// === === ===

void initNN()
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

                // get L-->R links
                if (x != 0) {
                   Neuron* nPrev = nn[line][x-1];
                   if (nPrev) {
                       Link* lnk = new Link(nPrev, nCur);
                       nCur->linksIn.append(lnk);
                       links.append(lnk);
                   }
                }

                // get L<--R links
                if (x != NN_W-1) {
                   Neuron* nNext = nn[line][x+1];
                   if (nNext) {
                       Link* lnk = new Link(nNext, nCur);
                       nCur->linksOut.append(lnk);
                   }
                }
            }
        }
    }

    for (int x = 0; x < NN_W; x++) {

        QVector<Neuron*> layer;

        for (int y = 0; y < NN_H; y++) {

            Neuron* n = nn[y][x];
            if (!n) { continue; }
            layer.append(n);
        }

        layersFwd.append(layer);
    }

    for (QVector<Neuron*> layer : layersFwd) { layersBck.append(layer); }
    std::reverse(layersBck.begin(), layersBck.end());
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
        //if (n == inputN || n == outputN) { continue; }
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

    // gather neuron weights

    for (Neuron* n : neurons) {
        if (n->weight < min) { min = n->weight; }
        if (n->weight > max) { max = n->weight; }
    }

    range = max - min;

    for (Neuron* n : neurons) {

        for (Link* l : n->linksIn) {

            int c = (l->neurFrom->weight - min) / range;

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

#define NUM_EXPERIMENTS     1000 // perform all experiments (values are the dataset)

#define INIT_WEIGHT_MIN -0.1
#define INIT_WEIGHT_MAX  0.1

bool trainingFinished = false;

void initWeightsRandom()
{
    for (Neuron* n : neurons) {
        if (n == inputN || n == outputN) { continue; }
        n->weight = fRand(INIT_WEIGHT_MIN, INIT_WEIGHT_MAX);
    }

     inputN->weight = 1.0;
    outputN->weight = 1.0;
}

void initWeights(double val)
{
    for (Neuron* n : neurons) { n->weight = val; }
}

void resetNN()
{
    for (Neuron* n : neurons) {
        n->outputCalculated = false;
        n->output = 0.0;
    }
}

void computeLayerForward(int id)
{
    for (Neuron* n : layersFwd[id]) {
        n->getOutputForward();
    }
}

void computeLayerBackward(int id)
{
    for (Neuron* n : layersBck[id]) {
        n->getOutputBackward();
    }
}

//double computeForward(double in)
//{
//    for (Neuron* n : neurons) { n->signalCalculated = false; }
//    inputN->setOutput(in);
//    return outputN->getOutputForward();
//}
//
//double computeBackward(double in)
//{
//    for (Neuron* n : neurons) { n->signalCalculated = false; }
//    outputN->setOutput(in);
//    return inputN->getOutputBackward();
//}

int main(int argc, char *argv[])
{
    simLoopSleepUs = 1000 * 1000;
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

    initNN();

    inputN ->isInput  = true;
    outputN->isOutput = true;

    //initWeightsRandom();
    initWeights(1.0);

    double inval = -10;

    while (running) {

        SDL_PumpEvents();
        processInputMainNN();

        //if (!upressed) { continue; }

// ======================================================================================
// ======================================================================================
// ======================================================================================

            printLayer(0);
            printLayer(1);
            printLayer(2);

//        inputN->setOutput(inval);
//        qDebug() << inval << computeForward(inval);
//
//        qDebug() << "-----------------------";
//        for (Neuron* n : neurons) { qDebug() << n->output; }
//        qDebug() << "-----------------------";
//
//        inval += 1;
//        if (inval > 10) { inval = -10; }

        //inputN->setOutput(1);


        //for (double v = -1; v < 1.0; v +=0.1) {
        //    qDebug() << nn[0][1]->activate(v);
        //}


//        qDebug() << "==========================================";
//        for (Neuron* n : neurons) { n->printLinksIn (); }
//        qDebug() << "==========================================";
//        std::reverse(neurons.begin(), neurons.end());
//        for (Neuron* n : neurons) { n->printLinksOut(); }
//        qDebug() << "==========================================";
//        for (Link* l : links) { l->print(); }

// ======================================================================================
// ======================================================================================
// ======================================================================================

        SDL_SetRenderDrawColor(renderer, 64, 64, 64, 0);
        SDL_RenderClear(renderer);

        renderNN(nn, NN_RENDER_OFFSET);

        renderFlag(0, trainingFinished ? 255 : 0, 0, 20, 60);

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
