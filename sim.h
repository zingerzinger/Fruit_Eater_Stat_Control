#ifndef SIM_H
#define SIM_H

#include <QDebug>
#include <QVector>

#include <QFile>
#include <QDataStream>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "defines.h"
#include "creature.h"

class Sim
{
public:
    Sim(SDL_Renderer* renderer);
    ~Sim();

    void step(double dt_secs);

    void render();

    void setVisualEnabled(bool enabled);

    void addCreature(int id, Creature &creature);
    void removeCreature(int id);

    void addFruitsRandom();
    void addFruit(Vec2 pos);

private:

    bool checkDistance(Vec2 f);

    QVector<Creature*> creatures;
    QVector<Vec2> fruits;

    bool enabled = true;
    SDL_Renderer* renderer = nullptr;

    void renderArc(Vec2 p, double r, double sAngleDegs, double arcLenDegs, double angleStepDegs);

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

    bool wrTelem = false;
    QFile* telemFile;
    QDataStream* telemStream;
};

Sim::Sim(SDL_Renderer* renderer)
{
    this->renderer = renderer;

#if SIM_WR_TELEM
    telemFile = new QFile("data.bin");
    if (telemFile->open(QIODevice::WriteOnly)) {
        wrTelem = true;
        telemStream = new QDataStream(telemFile);
    }
#endif
}

Sim::~Sim()
{
    if (!wrTelem) { return; }

    telemStream->commitTransaction();
    telemFile->close();

    delete telemStream;
    delete telemFile;
}

void Sim::renderArc(Vec2 p, double r, double sAngleDegs, double arcLenDegs, double angleStepDegs)
{
    Vec2 vp = addVec(p, rotateDegs(sAngleDegs, Vec2(r, 0)));
    for (double a = 0.0; a < arcLenDegs; a += angleStepDegs) {
        Vec2 vc = addVec(p, rotateDegs(sAngleDegs + a, Vec2(r, 0)));
        SDL_RenderDrawLine(renderer, vp.x, vp.y, vc.x, vc.y);
        vp = vc;
    }

    // last segment
    Vec2 vc = addVec(p, rotateDegs(sAngleDegs + arcLenDegs, Vec2(r, 0)));
    SDL_RenderDrawLine(renderer, vp.x, vp.y, vc.x, vc.y);
}

void Sim::render()
{
    if (!enabled) { return; }

    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 0); // green

    for (int i = 0; i < sizeof(wborder) / sizeof(Vec2) - 1; i++)
    {
        SDL_RenderDrawLine(renderer, wborder[i  ].x, wborder[i  ].y, wborder[i+1].x, wborder[i+1].y);
    }

    for (Creature* c : creatures) {
        for (int i = 0; i < sizeof(rcreature) / sizeof(Vec2) - 1; i++)
        {
            Vec2 va, vb;
            va = addVec(c->pos, rotateDegs(c->orientation, rcreature[i  ]));
            vb = addVec(c->pos, rotateDegs(c->orientation, rcreature[i+1]));
            SDL_RenderDrawLine(renderer, va.x, va.y, vb.x, vb.y);
        }
    }

    SDL_Rect foodRect;
    for (Creature* c : creatures) {
        foodRect.x = c->pos.x - (CREATURE_FOOD_MAX/2);
        foodRect.y = c->pos.y - (20);
        foodRect.w = c->food;
        foodRect.h = 5;

        SDL_SetRenderDrawColor(renderer,   0, 255,   0, 0); // green
        SDL_RenderFillRect(renderer, &foodRect);
        SDL_SetRenderDrawColor(renderer, 128, 128, 128, 0); // gray
        foodRect.w = CREATURE_FOOD_MAX;
        SDL_RenderDrawRect(renderer, &foodRect);
    }

    for (Creature* c : creatures) {
        // FOV, direction

        Vec2 vc;

        vc = addVec(c->pos, rotateDegs(c->orientation, Vec2(CREATURE_BEAM_R, 0)));
        SDL_RenderDrawLine(renderer, c->pos.x, c->pos.y, vc.x, vc.y);

        vc = addVec(c->pos, rotateDegs(c->orientation - CREATURE_FOV_DEGS*0.5, Vec2(CREATURE_BEAM_R, 0)));
        SDL_RenderDrawLine(renderer, c->pos.x, c->pos.y, vc.x, vc.y);

        vc = addVec(c->pos, rotateDegs(c->orientation + CREATURE_FOV_DEGS*0.5, Vec2(CREATURE_BEAM_R, 0)));
        SDL_RenderDrawLine(renderer, c->pos.x, c->pos.y, vc.x, vc.y);

        renderArc(c->pos, CREATURE_BEAM_R, c->orientation - CREATURE_FOV_DEGS*0.5, CREATURE_FOV_DEGS, 10.0);
    }

    SDL_Rect rect;
    rect.w = FRUIT_R+1;
    rect.h = FRUIT_R+1;
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 0); // red

    for (Vec2 f : fruits) {
        rect.x = f.x - (FRUIT_R/2);
        rect.y = f.y - (FRUIT_R/2);
        SDL_RenderFillRect(renderer, &rect);
    }

    for (Creature* c : creatures) {
        SDL_RenderDrawLine(renderer, c->pos.x, c->pos.y, c->target.x, c->target.y);
    }

    // render creature memory fruits
    SDL_SetRenderDrawColor(renderer, 255, 0, 255, 0); // magenta
    SDL_Rect rectMemFruit;
    rectMemFruit.w = FRUIT_R+6;
    rectMemFruit.h = FRUIT_R+6;

    for (Creature* c : creatures) {
        for (Vec2 f : c->fruits) {
            rectMemFruit.x = f.x - ((FRUIT_R+6)/2);
            rectMemFruit.y = f.y - ((FRUIT_R+6)/2);
            SDL_RenderDrawRect(renderer, &rectMemFruit);
        }
    }

    // trail
    SDL_SetRenderDrawColor(renderer, 128, 128, 128, 0); // grey
    for (Creature* c : creatures) {

        if (c->trail.size() < 2) { break; }

        Vec2 vp = c->trail.first();
        int i = 1;
        while (i < c->trail.size()) {
            Vec2 vc = c->trail[i];
            SDL_RenderDrawLine(renderer, vp.x, vp.y, vc.x, vc.y);
            vp = vc;
            i++;
        }
    }
}

void Sim::setVisualEnabled(bool enabled)
{

}

bool Sim::checkDistance(Vec2 fn)
{
    for (Vec2 f : fruits) {
        double distSq = VecLenSq(subVec(f, fn));
        if (distSq <= RND_FRUIT_SPACING*RND_FRUIT_SPACING) { return false; }
    }

    return true;
}

void Sim::addFruit(Vec2 pos)
{
    fruits.append(pos);
}

void Sim::addFruitsRandom()
{
    bool needBreak = false;

    while (true) {

        for (int i = 0; i < RND_GEN_ATTEMPS; i++) {
            Vec2 nf(fRand(0.0, W_W), fRand(0.0, W_H));
            needBreak = !checkDistance(nf);
            if (needBreak) { break; }
            fruits.append(nf);
        }

        if (needBreak || fruits.size() < RND_NUM_FRUITS) { break; }
    }
}

void Sim::step(double dt_secs)
{
    // creature physics
    for (Creature* c : creatures) {

        c->w   += (c->rotF / CREATURE_ANGM) * dt_secs;
        c->vel += (c->fwdF / CREATURE_MASS) * dt_secs;

        c->w   = clamp(c->w  , -CREATURE_MAX_ROT_SPEED_DPS, CREATURE_MAX_ROT_SPEED_DPS);
        c->vel = clamp(c->vel, -CREATURE_MAX_SPEED        , CREATURE_MAX_SPEED        );

        c->orientation += c->w * dt_secs;

        if (c->orientation > 360.0) { c->orientation -= 360.0; }
        if (c->orientation <   0.0) { c->orientation += 360.0; }

        Vec2 orient = rotateDegs(c->orientation, Vec2(1, 0));
        c->pos = addVec( c->pos, mulVecScalar(c->vel * dt_secs, orient) );

    }

    // creature - fruit interaction (no ray-sphere intersection yet)
    for (Creature* c : creatures) {

        c->food = clamp(c->food -= CREATURE_FOOD_HUNGER_PS * dt_secs, 0.0, CREATURE_FOOD_MAX);

        for (int i = fruits.size()-1; i >= 0; i--) {
            Vec2 f = fruits[i];

            Vec2 delta = subVec(f, c->pos);

            if (VecLenSq( delta ) <= ((CREATURE_R+FRUIT_R)*(CREATURE_R+FRUIT_R))) {
                c->removeFruit(f);
                c->food = clamp(c->food += SIM_FRUIT_FOOD, 0.0, CREATURE_FOOD_MAX);
                fruits.remove(i);
            }
        }
    }

    // show visible fruits
    for (Creature* c : creatures) {

        for (Vec2 f : fruits) {

            Vec2 delta = subVec(f, c->pos);

            if (VecLenSq(delta) > CREATURE_BEAM_R*CREATURE_BEAM_R) { continue; }

            Vec2 orientRayL = rotateDegs(c->orientation - CREATURE_FOV_DEGS * 0.5, Vec2(1, 0));
            Vec2 orientRayR = rotateDegs(c->orientation + CREATURE_FOV_DEGS * 0.5, Vec2(1, 0));

            if (vecCross(orientRayL, delta) < 0 ||
                vecCross(orientRayR, delta) > 0) { continue; }

            c->showFruit(f);
        }
    }

    for (Creature* c : creatures) { c->step(dt_secs); }

    // === === ===
    if (wrTelem) {
        Creature* cr = creatures.first();

        *telemStream << cr->targetChange
                     << cr->angleError
                     << cr->targetDistance
                     << cr->target.x
                     << cr->target.y
                     << cr->pos.x
                     << cr->pos.y
                     << cr->rotF
                     << cr->fwdF
                     << cr->vel
                     << cr->w;
    }
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
