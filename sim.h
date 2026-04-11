#ifndef SIM_H
#define SIM_H

#include <QDebug>
#include <QVector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "creature.h"

#define W_W 1280
#define W_H 1024

class Sim
{
public:
    Sim(SDL_Renderer* renderer);

    void step(double dt_secs);

    void render();

    void setVisualEnabled(bool enabled);

    void addCreature(int id, Creature &creature);
    void removeCreature(int id);

private:

    QVector<Creature*> creatures;

    bool enabled = true;
    SDL_Renderer* renderer = nullptr;

    // === graphics ===

    Vec2 wborder[5] = {
        {0,      0},
        {W_W-1,    0},
        {W_W-1,  W_H-1},
        {0,      W_H-1},
        {0,      0},
    };

    Vec2 rcreature[4] = {
        {-10,   10},
        { 10,    0},
        {-10,  -10},
        {-10,   10},
    };

    Vec2 rbeam[4] = {
        {10,    0},
        {80,   40},
        {80,  -40},
        {10,    0},
    };
};

Sim::Sim(SDL_Renderer* renderer)
{
    this->renderer = renderer;
}

void Sim::render()
{
    if (!enabled) { return; }

    for (int i = 0; i < sizeof(wborder) / sizeof(Vec2) - 1; i++)
    {
        SDL_RenderDrawLine(renderer, wborder[i  ].x, wborder[i  ].y, wborder[i+1].x, wborder[i+1].y);
    }

    for (Creature* c : creatures) {
        for (int i = 0; i < sizeof(rcreature) / sizeof(Vec2) - 1; i++)
        {
            Vec2 va, vb;

            va = addVec(c->pos, rotateDegs(c->angle, rcreature[i  ]));
            vb = addVec(c->pos, rotateDegs(c->angle, rcreature[i+1]));
            SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);

            va = addVec(c->pos, rotateDegs(c->angle, rbeam[i  ]));
            vb = addVec(c->pos, rotateDegs(c->angle, rbeam[i+1]));
            SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);
        }
    }

}

void Sim::setVisualEnabled(bool enabled)
{

}

void Sim::step(double dt_secs)
{
    for (Creature* c : creatures) {

        c->angle += c->rotSpeedDegs * dt_secs;
        if (c->angle > 360.0) { c->angle -= 360.0; }
        if (c->angle <   0.0) { c->angle += 360.0; }

        Vec2 orient = rotateDegs(c->angle, Vec2(1, 0));
        c->pos = addVec( c->pos, mulVecScalar(c->speedMps * dt_secs, orient) );


    }

    for (Creature* c : creatures) { c->step(dt_secs); }
}

void Sim::addCreature(int id, Creature &creature)
{
    creatures.append(&creature);
}

void Sim::removeCreature(int id)
{
    delete creatures[id];
    creatures.remove(id);
}

#endif // SIM_H

// ===========




//void renderSmoke()
//{
//    SDL_Rect rect;
//    rect.w = 4;
//    rect.h = 4;
//
//    SDL_SetRenderDrawColor(renderer, 172, 172, 172, 255);
//
//    for (Smoke s : smoke)
//    {
//        rect.x = s.pos.x - rect.w / 2;
//        rect.y = s.pos.y - rect.h / 2;
//
//        int r = 255;
//        int g = 255;
//        int b = 255;
//
//        if (s.life > 256-40) {
//            r = 255;
//            g = 255;
//            b = 0;
//        } else if (s.life > 256-72) {
//            r = s.life;
//            g = 128;
//            b = 128;
//        } else {
//            r = 52 + s.life;
//            g = 52 + s.life;
//            b = 52 + s.life;
//        }
//
//        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
//
//        SDL_RenderFillRect(renderer, &rect);
//    }
//}
//
//void renderShip()
//{
//    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
//
//    Vec2 va;
//    Vec2 vb;
//
//    for (int i = 0; i < sizeof(ship) / sizeof(Vec2) - 1; i++)
//    {
//        va = addVec(shipPos, rotateDegs(shipAngleDegs, ship[i  ]));
//        vb = addVec(shipPos, rotateDegs(shipAngleDegs, ship[i+1]));
//        SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);
//    }
//
//
//    for (int i = 0; i < sizeof(shipCenter) / sizeof(Vec2) - 1; i++)
//    {
//        va = addVec(shipPos, rotateDegs(shipAngleDegs, shipCenter[i  ]));
//        vb = addVec(shipPos, rotateDegs(shipAngleDegs, shipCenter[i+1]));
//        SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);
//    }
//
//    for (int i = 0; i < sizeof(shipRCS) / sizeof(Vec2) - 1; i++)
//    {
//        va = addVec(shipPos, rotateDegs(shipAngleDegs, shipRCS[i  ]));
//        vb = addVec(shipPos, rotateDegs(shipAngleDegs, shipRCS[i+1]));
//        SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);
//    }
//
//
//    Vec2 nozzlePosShip = {0, 39};
//
//    for (int i = 0; i < sizeof(shipEngine) / sizeof(Vec2) - 1; i++)
//    {
//        va = addVec(shipPos, rotateDegs(shipAngleDegs, addVec(nozzlePosShip, rotateDegs(gimbalAngleDegs, shipEngine[i  ]))));
//        vb = addVec(shipPos, rotateDegs(shipAngleDegs, addVec(nozzlePosShip, rotateDegs(gimbalAngleDegs, shipEngine[i+1]))));
//        SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);
//    }
//}
