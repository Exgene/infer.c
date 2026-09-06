#include "q4.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                    uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

void dequantize_row_q4_k(const BlockQ4_K *blocks, float *y, int k) {
  int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t *q = blocks[i].qs;
    float d = fp16_to_fp32(blocks[i].d);
    float min = fp16_to_fp32(blocks[i].dmin);
    int is = 0;

    for (int j = 0; j < QK_K; j += 64) {
      uint8_t sc, m;
      get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
      float d1 = d * sc;
      float m1 = min * m;
      get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
      float d2 = d * sc;
      float m2 = min * m;
      for (int l = 0; l < 32; l++)
        *y++ = d1 * (float)(q[l] & 0xF) - m1;
      for (int l = 0; l < 32; l++)
        *y++ = d2 * (float)(q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void dequantize_row_q6_k(const BlockQ6_K *blocks, float *y, int k) {
  int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    float d = fp16_to_fp32(blocks[i].d);
    const uint8_t *ql = blocks[i].ql;
    const uint8_t *qh = blocks[i].qh;
    const int8_t *sc = blocks[i].scales;

    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; l++) {
        int is = l / 16;
        int8_t q1 =
            (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int8_t q2 =
            (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int8_t q3 =
            (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int8_t q4 =
            (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void matvec_q4_k(float *y, const BlockQ4_K *blocks, const float *x, int out,
                 int in) {
  int blocks_per_row = in / QK_K;

#pragma omp parallel for
  for (int i = 0; i < out; i++) {
    const BlockQ4_K *row = blocks + (size_t)i * blocks_per_row;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
      float d = fp16_to_fp32(row[b].d);
      float dmin = fp16_to_fp32(row[b].dmin);
      const uint8_t *q = row[b].qs;
      int is = 0;
      int base = b * QK_K;
      for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, row[b].scales, &sc, &m);
        float d1 = d * sc;
        float m1 = dmin * m;
        get_scale_min_k4(is + 1, row[b].scales, &sc, &m);
        float d2 = d * sc;
        float m2 = dmin * m;
        for (int l = 0; l < 32; l++) {
          sum += (d1 * (float)(q[l] & 0xF) - m1) * x[base + j + l];
          sum += (d2 * (float)(q[l] >> 4) - m2) * x[base + j + 32 + l];
        }
        q += 32;
        is += 2;
      }
    }
    y[i] = sum;
  }
}

void matmul_q4_k(float *Y, const BlockQ4_K *blocks, const float *X, int n,
                 int out, int in) {
#pragma omp parallel for
  for (int t = 0; t < n; t++) {
    matvec_q4_k(Y + (size_t)t * out, blocks, X + (size_t)t * in, out, in);
  }
}

void matvec_q6_k(float *y, const BlockQ6_K *blocks, const float *x, int out,
                 int in) {
  int blocks_per_row = in / QK_K;

#pragma omp parallel for
  for (int i = 0; i < out; i++) {
    const BlockQ6_K *row = blocks + (size_t)i * blocks_per_row;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
      float d = fp16_to_fp32(row[b].d);
      const uint8_t *ql = row[b].ql;
      const uint8_t *qh = row[b].qh;
      const int8_t *sc = row[b].scales;
      int base = b * QK_K;
      for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
          int is = l / 16;
          int q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
          int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
          int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
          int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
          sum += d * sc[is + 0] * (q1 - 32) * x[base + n + l];
          sum += d * sc[is + 2] * (q2 - 32) * x[base + n + l + 32];
          sum += d * sc[is + 4] * (q3 - 32) * x[base + n + l + 64];
          sum += d * sc[is + 6] * (q4 - 32) * x[base + n + l + 96];
        }
        ql += 64;
        qh += 32;
        sc += 8;
      }
    }
    y[i] = sum;
  }
}

void matmul_q6_k(float *Y, const BlockQ6_K *blocks, const float *X, int n,
                 int out, int in) {
#pragma omp parallel for
  for (int t = 0; t < n; t++) {
    matvec_q6_k(Y + (size_t)t * out, blocks, X + (size_t)t * in, out, in);
  }
}
