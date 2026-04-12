#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string.h>
#include <chrono>
#include <math.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "defines.h"
#include "utils.h"
#include "sim.h"
#include "creature.h"

/* TODO:
 * ? statitics gathering system
 * ~ random fruit generation (same as roaming)
 * - NN training
 */

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

Sim* sim;
Creature* creature;

void processInput()
{
    SDL_GetMouseState(&mx, &my);

    SDL_Event event;
    while (SDL_PollEvent(&event)) {

        switch (event.type) {

            case SDL_MOUSEBUTTONDOWN: {

                int mX, mY;
                SDL_GetMouseState(&mX, &mY);
                sim->addFruit(Vec2(mX, mY));

            } break;

            case SDL_KEYDOWN: {
                if (event.key.keysym.sym == SDLK_ESCAPE) { running = false; }

                if (event.key.keysym.sym == SDLK_UP   ) { upressed =  true; }
                if (event.key.keysym.sym == SDLK_DOWN ) { dpressed =  true; }
                if (event.key.keysym.sym == SDLK_LEFT ) { lpressed =  true; }
                if (event.key.keysym.sym == SDLK_RIGHT) { rpressed =  true; }

                if (event.key.keysym.sym == SDLK_m) { creature->manual = !creature->manual; }

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

#include <QFile>
#include <QDataStream>
void testTelem()
{
    QFile* telemFile;
    QDataStream* telemStream;

    telemFile = new QFile("data_raw.bin");
    if (!telemFile->open(QIODevice::ReadOnly)) { return; }

    telemStream = new QDataStream(telemFile);

    bool   targetChange  ;
    double angleError    ;
    double targetDistance;
    double targetx       ;
    double targety       ;
    double posx          ;
    double posy          ;
    double rotF          ;
    double fwdF          ;
    double vel           ;
    double w;            ;

    int t;

    while (!telemStream->atEnd()) {
        *telemStream  >> targetChange  ;
        *telemStream  >> angleError    ;
        *telemStream  >> targetDistance;
        *telemStream  >> targetx       ;
        *telemStream  >> targety       ;
        *telemStream  >> posx          ;
        *telemStream  >> posy          ;
        *telemStream  >> rotF          ;
        *telemStream  >> fwdF          ;
        *telemStream  >> vel           ;
        *telemStream  >> w;            ;

        qDebug() << t
                 << targetChange
                 << angleError
                 << targetDistance
                 << targetx
                 << targety
                 << posx
                 << posy
                 << rotF
                 << fwdF
                 << vel
                 << w;
        t++;
    }
}

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

    // === === ===

    sim = new Sim(renderer);
    creature = new Creature();
    sim->addCreature(0, *creature);

    // === === ===

    uint64_t frameNum = 0;

    while (running) {

        SDL_PumpEvents();

        processInput();

        int ud = 0;
        int lr = 0;

        if (upressed) { ud += 1; }
        if (dpressed) { ud -= 1; }
        if (lpressed) { lr -= 1; }
        if (rpressed) { lr += 1; }

        creature->manualControl(ud * CREATURE_FWDFMAX, lr * CREATURE_ANGFMAX);

        sim->step(SIM_DT);

        if (frameNum % simSkipFrames == 0) {
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
            SDL_RenderClear(renderer);
            sim->render();
            SDL_RenderPresent(renderer);
        }

        std::this_thread::sleep_for(std::chrono::microseconds(simLoopSleepUs));

        frameNum++;
    }

    // === === ===

    delete sim;

    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
