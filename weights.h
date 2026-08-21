#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "config.h"
#include "safetensors.h"
#include <stdint.h>

typedef struct {
  float *rms_att;
  float *wq;
  float *wk;
  float *wv;
  float *wo;
  float *rms_ffn;
  float *w_gate;
  float *w_up;
  float *w_down;
} Layer;

typedef struct {
  float *token_emb;
  Layer layers[16];
  float *rms_final;
} Weights;

int bind_weights(const SafeTensors *, const WeightsConfigJson *, Weights *);

float *convert_tensor(const uint16_t *src, size_t n);

void free_weights(Weights *w, int num_layers);

#endif
