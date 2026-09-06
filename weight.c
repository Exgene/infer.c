#include "weight.h"

#include "q4.h"

#include <stdlib.h>
#include <string.h>

void weight_free(Weight *w) {
  if (w == NULL)
    return;
  if (w->owned && w->type == WT_F32)
    free(w->data.f32);
  memset(w, 0, sizeof(*w));
}

Weight weight_make_f32(float *data, int rows, int cols, int owned) {
  Weight w = {0};
  w.type = WT_F32;
  w.rows = rows;
  w.cols = cols;
  w.data.f32 = data;
  w.owned = owned;
  return w;
}

Weight weight_make_q4_k(const BlockQ4_K *blocks, int rows, int cols) {
  Weight w = {0};
  w.type = WT_Q4_K;
  w.rows = rows;
  w.cols = cols;
  w.data.q4_k = blocks;
  w.owned = 0;
  return w;
}

Weight weight_make_q6_k(const BlockQ6_K *blocks, int rows, int cols) {
  Weight w = {0};
  w.type = WT_Q6_K;
  w.rows = rows;
  w.cols = cols;
  w.data.q6_k = blocks;
  w.owned = 0;
  return w;
}

void weight_lookup_row(float *out, const Weight *w, int row) {
  if (w->type == WT_F32) {
    memcpy(out, w->data.f32 + (size_t)row * w->cols,
           (size_t)w->cols * sizeof(float));
    return;
  }

  if (w->type == WT_Q4_K) {
    int blocks_per_row = w->cols / QK_K;
    dequantize_row_q4_k(w->data.q4_k + (size_t)row * blocks_per_row, out,
                        w->cols);
    return;
  }

  if (w->type == WT_Q6_K) {
    int blocks_per_row = w->cols / QK_K;
    dequantize_row_q6_k(w->data.q6_k + (size_t)row * blocks_per_row, out,
                        w->cols);
  }
}
