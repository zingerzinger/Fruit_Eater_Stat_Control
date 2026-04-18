#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string.h>
#include <chrono>
#include <math.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tensorflow/include/tensorflow/core/public/session.h>
#include <tensorflow/c/c_api.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <QDebug>

#include "defines.h"
#include "utils.h"

using namespace std;
using namespace tensorflow;

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

void renderFlag(int r, int g, int b, int x, int y)
{
    SDL_Rect rect;
    rect.w = 20;
    rect.h = 20;

    SDL_SetRenderDrawColor(renderer, r, g, b, 0);
    rect.x = x;
    rect.y = y;
    SDL_RenderFillRect(renderer, &rect);
}

// See! https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2

typedef struct model_t {
  TF_Graph* graph;
  TF_Session* session;
  TF_Status* status;

  TF_Output input, target, output;

  TF_Operation *init_op, *train_op, *save_op, *restore_op;
  TF_Output checkpoint_file;
} model_t;

int Okay(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: %s\n", TF_Message(status));
    return 0;
  }
  return 1;
}

TF_Buffer* ReadFile(const char* filename) {
  int fd = open(filename, 0);
  if (fd < 0) {
    perror("failed to open file: ");
    return NULL;
  }
  struct stat stat;
  if (fstat(fd, &stat) != 0) {
    perror("failed to read file: ");
    return NULL;
  }
  char* data = (char*)malloc(stat.st_size);
  ssize_t nread = read(fd, data, stat.st_size);
  if (nread < 0) {
    perror("failed to read file: ");
    free(data);
    return NULL;
  }
  if (nread != stat.st_size) {
    fprintf(stderr, "read %zd bytes, expected to read %zd\n", nread,
            stat.st_size);
    free(data);
    return NULL;
  }
  TF_Buffer* ret = TF_NewBufferFromString(data, stat.st_size);
  free(data);
  return ret;
}

int ModelCreate(model_t* model, const char* graph_def_filename) {
  model->status = TF_NewStatus();
  model->graph = TF_NewGraph();

  {
    // Create the session.
    TF_SessionOptions* opts = TF_NewSessionOptions();
    model->session = TF_NewSession(model->graph, opts, model->status);
    TF_DeleteSessionOptions(opts);
    if (!Okay(model->status)) return 0;
  }

  TF_Graph* g = model->graph;

  {
    // Import the graph.
    TF_Buffer* graph_def = ReadFile(graph_def_filename);
    if (graph_def == NULL) return 0;
    printf("Read GraphDef of %zu bytes\n", graph_def->length);
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(g, graph_def, opts, model->status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_def);
    if (!Okay(model->status)) return 0;
  }

  // Handles to the interesting operations in the graph.
  model->input.oper = TF_GraphOperationByName(g, "input");
  model->input.index = 0;
  model->target.oper = TF_GraphOperationByName(g, "target");
  model->target.index = 0;
  model->output.oper = TF_GraphOperationByName(g, "output");
  model->output.index = 0;

  model->init_op = TF_GraphOperationByName(g, "init");
  model->train_op = TF_GraphOperationByName(g, "train");
  model->save_op = TF_GraphOperationByName(g, "save/control_dependency");
  model->restore_op = TF_GraphOperationByName(g, "save/restore_all");

  model->checkpoint_file.oper = TF_GraphOperationByName(g, "save/Const");
  model->checkpoint_file.index = 0;
  return 1;
}

int ModelInit(model_t* model) {
  const TF_Operation* init_op[1] = {model->init_op};
  TF_SessionRun(model->session, NULL,
                /* No inputs */
                NULL, NULL, 0,
                /* No outputs */
                NULL, NULL, 0,
                /* Just the init operation */
                init_op, 1,
                /* No metadata */
                NULL, model->status);
  return Okay(model->status);
}

void tf_experiments()
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    qDebug() << ( status.ok() ? "Session successfully created" : "Session creation ERROR" );

    const char* graph_def_filename = "graph.pb";
    const char* checkpoint_prefix = "./checkpoints/checkpoint";
//    int restore = DirectoryExists("checkpoints");

    int restore = 0;

    model_t model;
    qDebug() <<  "Loading graph";
    if (!ModelCreate(&model, graph_def_filename)) { qDebug() << "ModelCreate error"; };
    if (restore) {
    //printf(
    //    "Restoring weights from checkpoint (remove the checkpoints directory "
    //    "to reset)\n");
    //if (!ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) return 1;
    } else {
        qDebug() << "Initializing model weights";
        if (!ModelInit(&model)) { qDebug() << "ModelInit error"; };
    }

}

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

    // === init tensorflow ===

    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) { cout << status.ToString() << "\n"; return 1; }
    qDebug() << "Session successfully created.\n";

    tf_experiments();

    // === === ===

    uint64_t frameNum = 0;

    while (running) {

        SDL_PumpEvents();
        processInputMainNN();

        //if (!upressed) { continue; }

// ======================================================================================
// ======================================================================================
// ======================================================================================



// ======================================================================================
// ======================================================================================
// ======================================================================================

        SDL_SetRenderDrawColor(renderer, 64, 64, 64, 0);
        SDL_RenderClear(renderer);

        renderFlag(0, 0, 0, 20, 60);

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
