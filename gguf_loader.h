#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include "config.h"
#include "weights.h"

int gguf_loader_open(const char *path, Weights *w, const WeightsConfigJson *cfg);

#endif
