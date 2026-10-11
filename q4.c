#include "q4.h"

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

#ifdef __AVX2__
static inline float horizontal_sum(__m256 v) {
  __m128 low = _mm256_castps256_ps128(v);
  __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

static inline float dot_q4_k_block(const BlockQ4_K *block, const float *x) {
  float d = fp16_to_fp32(block->d);
  float dmin = fp16_to_fp32(block->dmin);
  const uint8_t *q = block->qs;
  __m256 sum = _mm256_setzero_ps();
  int is = 0;

  for (int j = 0; j < QK_K; j += 64) {
    uint8_t sc, m;
    get_scale_min_k4(is + 0, block->scales, &sc, &m);
    __m256 d1 = _mm256_set1_ps(d * sc);
    __m256 m1 = _mm256_set1_ps(dmin * m);
    get_scale_min_k4(is + 1, block->scales, &sc, &m);
    __m256 d2 = _mm256_set1_ps(d * sc);
    __m256 m2 = _mm256_set1_ps(dmin * m);

    for (int l = 0; l < 32; l += 8) {
      __m128i bytes = _mm_loadl_epi64((const __m128i *)(q + l));
      __m256i values = _mm256_cvtepu8_epi32(bytes);
      __m256 low = _mm256_cvtepi32_ps(
          _mm256_and_si256(values, _mm256_set1_epi32(0xF)));
      __m256 high =
          _mm256_cvtepi32_ps(_mm256_srli_epi32(values, 4));
      __m256 w1 = _mm256_sub_ps(_mm256_mul_ps(low, d1), m1);
      __m256 w2 = _mm256_sub_ps(_mm256_mul_ps(high, d2), m2);
      sum = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + j + l), sum);
      sum = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + j + l + 32), sum);
    }
    q += 32;
    is += 2;
  }

  return horizontal_sum(sum);
}

static inline float dot_q6_k_block(const BlockQ6_K *block, const float *x) {
  float d = fp16_to_fp32(block->d);
  const uint8_t *ql = block->ql;
  const uint8_t *qh = block->qh;
  const int8_t *sc = block->scales;
  const __m256i mask3 = _mm256_set1_epi32(3);
  const __m256i mask15 = _mm256_set1_epi32(0xF);
  const __m256i bias = _mm256_set1_epi32(32);
  __m256 sum = _mm256_setzero_ps();

  for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; l += 8) {
      int is = l / 16;
      __m256i lo1 = _mm256_cvtepu8_epi32(
          _mm_loadl_epi64((const __m128i *)(ql + l)));
      __m256i lo2 = _mm256_cvtepu8_epi32(
          _mm_loadl_epi64((const __m128i *)(ql + l + 32)));
      __m256i hi = _mm256_cvtepu8_epi32(
          _mm_loadl_epi64((const __m128i *)(qh + l)));

      __m256i q1 = _mm256_or_si256(
          _mm256_and_si256(lo1, mask15),
          _mm256_slli_epi32(_mm256_and_si256(hi, mask3), 4));
      __m256i q2 = _mm256_or_si256(
          _mm256_and_si256(lo2, mask15),
          _mm256_slli_epi32(
              _mm256_and_si256(_mm256_srli_epi32(hi, 2), mask3), 4));
      __m256i q3 = _mm256_or_si256(
          _mm256_srli_epi32(lo1, 4),
          _mm256_slli_epi32(
              _mm256_and_si256(_mm256_srli_epi32(hi, 4), mask3), 4));
      __m256i q4 = _mm256_or_si256(
          _mm256_srli_epi32(lo2, 4),
          _mm256_slli_epi32(
              _mm256_and_si256(_mm256_srli_epi32(hi, 6), mask3), 4));

      __m256 w1 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(q1, bias)),
          _mm256_set1_ps(d * sc[is + 0]));
      __m256 w2 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(q2, bias)),
          _mm256_set1_ps(d * sc[is + 2]));
      __m256 w3 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(q3, bias)),
          _mm256_set1_ps(d * sc[is + 4]));
      __m256 w4 = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_sub_epi32(q4, bias)),
          _mm256_set1_ps(d * sc[is + 6]));

      sum = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + n + l), sum);
      sum = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + n + l + 32), sum);
      sum = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + n + l + 64), sum);
      sum = _mm256_fmadd_ps(w4, _mm256_loadu_ps(x + n + l + 96), sum);
    }
    ql += 64;
    qh += 32;
    sc += 8;
  }

  return horizontal_sum(sum);
}
#endif

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
#ifdef __AVX2__
      const __m256i mask15 = _mm256_set1_epi32(0xF);
      __m256 vd1 = _mm256_set1_ps(d1);
      __m256 vm1 = _mm256_set1_ps(m1);
      __m256 vd2 = _mm256_set1_ps(d2);
      __m256 vm2 = _mm256_set1_ps(m2);
      for (int l = 0; l < 32; l += 8) {
        __m256i values =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(q + l)));
        __m256 low = _mm256_cvtepi32_ps(_mm256_and_si256(values, mask15));
        __m256 high = _mm256_cvtepi32_ps(_mm256_srli_epi32(values, 4));
        _mm256_storeu_ps(y + l, _mm256_fmsub_ps(low, vd1, vm1));
        _mm256_storeu_ps(y + l + 32, _mm256_fmsub_ps(high, vd2, vm2));
      }
      y += 64;
#else
      for (int l = 0; l < 32; l++)
        *y++ = d1 * (float)(q[l] & 0xF) - m1;
      for (int l = 0; l < 32; l++)
        *y++ = d2 * (float)(q[l] >> 4) - m2;
#endif
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
#ifdef __AVX2__
      const __m256i mask3 = _mm256_set1_epi32(3);
      const __m256i mask15 = _mm256_set1_epi32(0xF);
      const __m256i bias = _mm256_set1_epi32(32);
      for (int l = 0; l < 32; l += 8) {
        int is = l / 16;
        __m256i lo1 = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64((const __m128i *)(ql + l)));
        __m256i lo2 = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64((const __m128i *)(ql + l + 32)));
        __m256i hi = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64((const __m128i *)(qh + l)));

        __m256i q1 = _mm256_or_si256(
            _mm256_and_si256(lo1, mask15),
            _mm256_slli_epi32(_mm256_and_si256(hi, mask3), 4));
        __m256i q2 = _mm256_or_si256(
            _mm256_and_si256(lo2, mask15),
            _mm256_slli_epi32(
                _mm256_and_si256(_mm256_srli_epi32(hi, 2), mask3), 4));
        __m256i q3 = _mm256_or_si256(
            _mm256_srli_epi32(lo1, 4),
            _mm256_slli_epi32(
                _mm256_and_si256(_mm256_srli_epi32(hi, 4), mask3), 4));
        __m256i q4 = _mm256_or_si256(
            _mm256_srli_epi32(lo2, 4),
            _mm256_slli_epi32(
                _mm256_and_si256(_mm256_srli_epi32(hi, 6), mask3), 4));

        _mm256_storeu_ps(y + l,
                         _mm256_mul_ps(_mm256_set1_ps(d * sc[is + 0]),
                                       _mm256_cvtepi32_ps(
                                           _mm256_sub_epi32(q1, bias))));
        _mm256_storeu_ps(y + l + 32,
                         _mm256_mul_ps(_mm256_set1_ps(d * sc[is + 2]),
                                       _mm256_cvtepi32_ps(
                                           _mm256_sub_epi32(q2, bias))));
        _mm256_storeu_ps(y + l + 64,
                         _mm256_mul_ps(_mm256_set1_ps(d * sc[is + 4]),
                                       _mm256_cvtepi32_ps(
                                           _mm256_sub_epi32(q3, bias))));
        _mm256_storeu_ps(y + l + 96,
                         _mm256_mul_ps(_mm256_set1_ps(d * sc[is + 6]),
                                       _mm256_cvtepi32_ps(
                                           _mm256_sub_epi32(q4, bias))));
      }
#else
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
#endif
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
#ifdef __AVX2__
      sum += dot_q4_k_block(&row[b], x + b * QK_K);
#else
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
#endif
    }
    y[i] = sum;
  }
}

// Batched over tokens: walk the weights once, dequantize a tile of rows into a
// float buffer and let sgemm multiply it with every token, instead of one full
// matvec (a full pass over the matrix) per token. 32 rows measured best: larger
// tiles fall out of cache for wide matrices when all threads are busy.
#define MATMUL_TILE_ROWS 32

void matmul_q4_k(float *Y, const BlockQ4_K *blocks, const float *X, int n,
                 int out, int in) {
  int blocks_per_row = in / QK_K;

#pragma omp parallel
  {
    float *tile = malloc((size_t)MATMUL_TILE_ROWS * in * sizeof(float));
    if (!tile) {
      perror("malloc");
      abort();
    }

#pragma omp for schedule(static)
    for (int i0 = 0; i0 < out; i0 += MATMUL_TILE_ROWS) {
      int rows = out - i0 < MATMUL_TILE_ROWS ? out - i0 : MATMUL_TILE_ROWS;
      dequantize_row_q4_k(blocks + (size_t)i0 * blocks_per_row, tile, rows * in);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, rows, in, 1.0f,
                  X, in, tile, in, 0.0f, Y + i0, out);
    }

    free(tile);
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
#ifdef __AVX2__
      sum += dot_q6_k_block(&row[b], x + b * QK_K);
#else
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
#endif
    }
    y[i] = sum;
  }
}

void matmul_q6_k(float *Y, const BlockQ6_K *blocks, const float *X, int n,
                 int out, int in) {
  int blocks_per_row = in / QK_K;

#pragma omp parallel
  {
    float *tile = malloc((size_t)MATMUL_TILE_ROWS * in * sizeof(float));
    if (!tile) {
      perror("malloc");
      abort();
    }

#pragma omp for schedule(static)
    for (int i0 = 0; i0 < out; i0 += MATMUL_TILE_ROWS) {
      int rows = out - i0 < MATMUL_TILE_ROWS ? out - i0 : MATMUL_TILE_ROWS;
      dequantize_row_q6_k(blocks + (size_t)i0 * blocks_per_row, tile, rows * in);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, rows, in, 1.0f,
                  X, in, tile, in, 0.0f, Y + i0, out);
    }

    free(tile);
  }
}
