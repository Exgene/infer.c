#ifndef LOADER_H
#define LOADER_H

#include "config.h"
#include "weights.h"

typedef enum {
  LOADER_SAFETENSORS = 0,
  LOADER_GGUF,
  LOADER_UNKNOWN,
} LoaderType;

LoaderType loader_detect(const char *path);
int loader_open(const char *path, Weights *w, const WeightsConfigJson *cfg);
void loader_close(Weights *w, int num_layers);

#endif
