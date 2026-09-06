#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "config.h"
#include "weight.h"
#include <stdint.h>

typedef struct {
  float *rms_att;
  float *rms_ffn;
  Weight wq;
  Weight wk;
  Weight wv;
  Weight wo;
  Weight w_gate;
  Weight w_up;
  Weight w_down;
} Layer;

typedef struct {
  Weight token_emb;
  Layer layers[16];
  float *rms_final;
  void *backing;
  int qk_interleaved;
} Weights;

void free_weights(Weights *w, int num_layers);

#endif
