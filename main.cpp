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
    Link(Neuron* neuronFrom, Neuron* neuronTo, double weight)
    {
        this->weight = weight;
        this->neurFrom = neuronFrom;
        this->neurTo   = neuronTo;
    }

    double weight;

    Neuron* neurFrom;
    Neuron* neurTo;

    void print();
};

#define TAN_ONE (1 / 1.5575)

#define NUM_EXPERIMENTS     1000 // perform all experiments (values are the dataset)

#define INIT_WEIGHT_MIN -0.1
#define INIT_WEIGHT_MAX  0.1

#define WEIGHT_DELTA_NOISE_MIN -0.01
#define WEIGHT_DELTA_NOISE_MAX  0.01

bool trainingFinished = false;

struct Neuron
{
    bool outputCalculated = false;
    double output = 0;

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
             out += link->neurFrom->getOutputForward() * link->weight;
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
             out += link->neurTo->getOutputBackward() * link->weight;
         }

         outputCalculated = true;
         output = activate(out);
         return output;
    }

    double activate(double val) {
        // sigmoid : 1 / (1 + e^-x)
        return 1.0 / (1.0 + pow(M_E, -val));
    }

    double derivative(double val) {
        return val * (1.0 - val);
    }

//    void updateInLinkWeights(double error) {
//        double totalLinksInWeight = 0.0;
//        double min =  MAXFLOAT;
//        double max = -MAXFLOAT;
//
//        for (Link* l : linksIn) {
//            if (l->weight < min) { min = l->weight; }
//            if (l->weight > max) { max = l->weight; }
//        }
//
//        for (Link* l : linksIn) {
//            double k = (l->weight - min) / (max - min);
//            if (linksIn.size() == 1) { k = 1.0; }
//            l->weight += error * k + fRand(WEIGHT_DELTA_NOISE_MIN, WEIGHT_DELTA_NOISE_MAX);
//        }
//    }

    void updateInLinkWeights(double error) {
        for (Link* l : linksIn) {
            l->weight += derivative(error) + fRand(WEIGHT_DELTA_NOISE_MIN, WEIGHT_DELTA_NOISE_MAX);
        }
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
        qDebug() << ( QString("[%1][%2] | %3").arg(yid)
                                              .arg(xid)
                                              .arg(output));
    }

    void printLinksIn () { for (Link* l : linksIn ) { l->print(); } }
    void printLinksOut() { for (Link* l : linksOut) { l->print(); } }
};

void Link::print() {
    qDebug() << ( QString("[%1][%2] --> [%3][%4] | %5").arg(neurFrom->yid)
                                                       .arg(neurFrom->xid)
                                                       .arg(neurTo  ->yid)
                                                       .arg(neurTo  ->xid)
                                                       .arg(weight));
}

#define NN_H 3
#define NN_W 3

//// ------------------------------------------------------------------------------------------//
////                                       in    -->      n    --     out                      //
//Neuron* nn[NN_H][NN_W] = {{ /* --> */ new Neuron(), new Neuron(), new Neuron() /* --> */},}; //
//// ------------------------------------------------------------------------------------------//

// --------------------------------------------------------------------------------------------
//                                       in    -->      n    -->     out                     //
Neuron* nn[NN_H][NN_W] = {{ /* --> */ new Neuron(), new Neuron(), new Neuron() /* --> */},   //
                          {                      0, new Neuron(),                      0},   //
                          {                      0, new Neuron(),                      0},}; //
// --------------------------------------------------------------------------------------------

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
    // init neurons
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

                // init linksIn (input --> output)
                if (x != 0) {
                   Neuron* nPrev = nn[line][x-1];
                   if (nPrev) {
                       Link* lnk = new Link(nPrev, nCur, 0.0);
                       nCur ->linksIn .append(lnk);
                       nPrev->linksOut.append(lnk);
                       links.append(lnk);
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

    // gather link weights

    for (Link* l : links) {
        if (l->weight < min) { min = l->weight; }
        if (l->weight > max) { max = l->weight; }
    }

    range = max - min;

    for (Neuron* n : neurons) {

        for (Link* l : n->linksIn) {

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

void initWeights(double val)
{
    for (Link* l : links) { l->weight = val; }
}

void initWeightsRandom()
{
    for (Link* l : links) {
         l->weight = fRand(INIT_WEIGHT_MIN, INIT_WEIGHT_MAX);
    }
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

// find minimum (function roots) numerically

#define GRAD_STEP 0.00001
#define NUM_STEPS 1000

#define DESCENT_EPSILON 0.001

double f(QVector<double> x)
{
    return x[0]*x[0] + x[1]*x[1];
}

QVector<double> grad(QVector<double> x) {

    QVector<double> result;

    for (int i = 0; i < x.size(); i++) {
        QVector<double> tplus = x;
        QVector<double> tmins = x;

        tplus[i] += GRAD_STEP;
        tmins[i] -= GRAD_STEP;

        double g = f(tplus) - f(tmins);

        g /= 2 * GRAD_STEP;

        result.append(g); ;
    }

    // compute unit vector
    double vlen = 0.0;
    for (double v : result) { vlen += v*v; }
    double invLen = 1.0 / sqrt(vlen);
    for (int i = 0; i < result.size(); i++) { result[i] = result[i] * invLen; }
    return result;
}

void grad()
{
    // f = x^2 + y^2

    double x = 1;
    double y = 1;

    QVector<double> grd = grad(QVector<double>({x,y}));

    double gx = grd[0];
    double gy = grd[1];

    qDebug() << gx << gy;

    double s1;
    double s2;
    double s3;

    s1 = s2 = s3 = 0;

    while (1) {

        double func = f(QVector<double>({x,y}));

        s1 = s2;
        s2 = s3;
        s3 = func;

        if (s1 > s2 && s3 > s2) {
            gx *= -0.9;
            gy *= -0.9;
        }

        qDebug() << func << x << y;
        x -= gx;
        y -= gy;

        if (abs(func) <= DESCENT_EPSILON) { break; }
        std::this_thread::sleep_for(std::chrono::microseconds(50000));
    }

    qDebug() << "============";
    while (1) { std::this_thread::sleep_for(std::chrono::microseconds(1000000)); }
    // change 1 var
    // compute f

    // change 2 var
    // compute f

    //
}

int main(int argc, char *argv[])
{
    simLoopSleepUs = 100 ;//1000 * 1000;
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

    initWeights(1.0);
    initWeightsRandom();

    double inval = -10;

    int exp = 0;

    grad();

    while (running) {

        SDL_PumpEvents();
        processInputMainNN();

        //if (!upressed) { continue; }

// ======================================================================================
// ======================================================================================
// ======================================================================================

//        double input = exp / (double)NUM_EXPERIMENTS;
//        double target = exp / (double)NUM_EXPERIMENTS;

        double input  = 1;
        double target = 0.2;

        resetNN();

        inputN->setOutput(input);
        computeLayerForward(0);
        computeLayerForward(1);
        computeLayerForward(2);

        double output = outputN->output;

        double err = (target - output)*(target - output);

        for (Neuron* n : layersFwd[1]) {
            n->updateInLinkWeights(-err * 0.01);
        }

        qDebug() << output << err << nn[0][1]->linksIn[0]->weight;

        qDebug() << "==========================================";
        for (Neuron* n : neurons) { n->printLinksIn (); }
        qDebug() << "==========================================";
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
