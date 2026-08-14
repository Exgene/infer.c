#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "config.h"
#include "safetensors.h"
#include <stdint.h>

typedef struct {
  const uint16_t *rms_att;
  const uint16_t *wq;
  const uint16_t *wk;
  const uint16_t *wv;
  const uint16_t *wo;
  const uint16_t *rms_ffn;
  const uint16_t *w_gate;
  const uint16_t *w_up;
  const uint16_t *w_down;
} Layer;

typedef struct {
  const uint16_t *token_emb;
  Layer layers[16];
  const uint16_t *rms_final;
} Weights;

int bind_weights(const SafeTensors *, const WeightsConfigJson *, Weights *);

#endif
