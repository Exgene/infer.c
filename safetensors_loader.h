#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#include "config.h"
#include "weights.h"

int safetensors_loader_open(const char *path, Weights *w,
                            const WeightsConfigJson *cfg);

#endif
