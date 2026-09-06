#ifndef WEIGHT_H
#define WEIGHT_H

#include <stddef.h>
#include <stdint.h>

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qs[QK_K / 2];
} BlockQ4_K;

typedef struct {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  uint16_t d;
} BlockQ6_K;

typedef enum {
  WT_NONE = 0,
  WT_F32,
  WT_Q4_K,
  WT_Q6_K,
} WeightType;

typedef struct {
  WeightType type;
  int rows;
  int cols;
  union {
    float *f32;
    const BlockQ4_K *q4_k;
    const BlockQ6_K *q6_k;
  } data;
  int owned;
} Weight;

void weight_free(Weight *w);
Weight weight_make_f32(float *data, int rows, int cols, int owned);
Weight weight_make_q4_k(const BlockQ4_K *blocks, int rows, int cols);
Weight weight_make_q6_k(const BlockQ6_K *blocks, int rows, int cols);
void weight_lookup_row(float *out, const Weight *w, int row);

#endif
