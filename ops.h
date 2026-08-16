#ifndef OPS_H
#define OPS_H

#include "config.h"
#include "weights.h"
#include <stdint.h>

float bf16_to_float32(uint16_t in);

void lookup(float *x, const uint16_t *token_emb, int token_id, int hidden);

void rmsnorm(float *xn, const float *x, const uint16_t *weight, int n,
             float eps);

void matvec(float *y, const uint16_t *W, const float *x, int out, int in);

void add(float *x, const float *branch, int n);

void silu(float *x, int n);

void attention(float *out, const float *q, const float *k, const float *v,
               int seq_len, int head_dim, int num_kv_heads, int num_heads);

int forward(const WeightsConfigJson *cfg, const Weights *w, float *x, float *xn,
            float *q, float *k, float *v, float *attn, float *hb, float *hb2,
            float *logits, int token_id);

#endif
