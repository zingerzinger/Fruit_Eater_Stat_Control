#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string.h>
#include <chrono>
#include <math.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

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

                if (event.key.keysym.sym == SDLK_q) {
                    simLoopSleepUs += SIM_LOOP_SLEEP_STEP_US;
                    simSkipFrames--;

                    if (simSkipFrames <= 0) {
                        simLoopSleepUs = SIM_LOOP_SLEEP_STEP_US;
                        simSkipFrames  = SIM_SKIP_RENDER_FRAMES;
                    }
                }

                if (event.key.keysym.sym == SDLK_w) {
                    simLoopSleepUs -= SIM_LOOP_SLEEP_STEP_US;
                    simSkipFrames++;

                    if (simLoopSleepUs <= 0) {
                        simLoopSleepUs = SIM_LOOP_SLEEP_STEP_US;
                        simSkipFrames  = SIM_SKIP_RENDER_FRAMES;
                    }
                }

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

#define NN_NEURON_SIDE 20
#define NN_RENDER_SPACING 40
#define NN_RENDER_OFFSET 100

#define ACTIVATION_THRESHOLD 0.2

Neuron n;

Neuron* nn[NN_H][NN_W] = {{            0, new Neuron(), new Neuron(), 0            },
                          { new Neuron(), new Neuron(), new Neuron(), new Neuron() },
                          {            0, new Neuron(), new Neuron(), 0            },};

// === === ===

void initNNLinks()
{
    for (int y = 0; y < NN_H; y++) {
        for (int x = 0; x < NN_W; x++) {

            Neuron* nCur = nn[y][x];
            if (!nCur) { continue; }

            nCur->xRendCoord = x * NN_RENDER_SPACING + NN_RENDER_OFFSET;
            nCur->yRendCoord = y * NN_RENDER_SPACING + NN_RENDER_OFFSET;

            for (int line = 0; line < NN_H; line++) {

                if (x == 0) { continue; }
                Neuron* nPrev = nn[line][x-1];
                if (!nPrev) { continue; }

                Link* lnk = new Link(nPrev, nCur, 1.0);
                nCur->links.append(lnk);
            }
        }
    }
}

// === === ===

int main(int argc, char *argv[])
{
    //testTelem(); return 0;

    simLoopSleepUs = SIM_LOOP_SLEEP_STEP_US;
    simSkipFrames  = SIM_SKIP_RENDER_FRAMES;

    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1" );

    window = SDL_CreateWindow("main", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W_W, W_H, SDL_WINDOW_BORDERLESS | SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, 0);

    TTF_Init();
    font = TTF_OpenFont("Monoid-Bold.ttf", 12);
    if (font == NULL) { fprintf(stderr, "error: font not found\n"); exit(EXIT_FAILURE); }

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    uint64_t frameNum = 0;

    initNNLinks();

    while (running) {

        SDL_PumpEvents();

        processInputMainNN();

        int ud = 0;
        int lr = 0;

        if (upressed) { ud += 1; }
        if (dpressed) { ud -= 1; }
        if (lpressed) { lr -= 1; }
        if (rpressed) { lr += 1; }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
        SDL_RenderClear(renderer);

        SDL_Rect neuronVisRect;
        neuronVisRect.w = NN_NEURON_SIDE;
        neuronVisRect.h = NN_NEURON_SIDE;

        // render NN

        for (int y = 0; y < NN_H; y++) {
            for (int x = 0; x < NN_W; x++) {

                Neuron* n = nn[y][x];
                if (!n) { continue; }

                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 0);
                neuronVisRect.x = x * NN_RENDER_SPACING - (NN_NEURON_SIDE / 2) + NN_RENDER_OFFSET;
                neuronVisRect.y = y * NN_RENDER_SPACING - (NN_NEURON_SIDE / 2) + NN_RENDER_OFFSET;
                SDL_RenderDrawRect(renderer, &neuronVisRect);

                for (Link* l : n->links) {

                    SDL_SetRenderDrawColor(renderer, 0, 0, l->weight * 255.0, 0);

                    SDL_RenderDrawLine(renderer, l->neurFrom->xRendCoord, l->neurFrom->yRendCoord,
                                                 l->neurTo  ->xRendCoord, l->neurTo  ->yRendCoord);
                }
            }
        }

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
