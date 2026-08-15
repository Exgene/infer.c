#ifndef OPS_H
#define OPS_H

#include <stdint.h>

float bf16_to_float32(uint16_t in);

void lookup(float *x, const uint16_t *token_emb, int token_id, int hidden);

void rmsnorm(float *xn, const float *x, const uint16_t *weight, int n,
             float eps);

void matvec(float *y, const uint16_t *W, const float *x, int out, int in);

void add(float *x, const float *branch, int n);

void silu(float *x, int n);

#endif
