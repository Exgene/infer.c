#include "q4.h"

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static inline float horizontal_sum128(__m128 v) {
  v = _mm_hadd_ps(v, v);
  v = _mm_hadd_ps(v, v);
  return _mm_cvtss_f32(v);
}

// Unpack Q4_K's 12 packed bytes into 16 int16: the 8 scales in the low half,
// the 8 mins in the high half. The result is built from registers on purpose:
// writing utmp to the stack and loading it back as one vector stalls on store
// forwarding (measured ~35% of the decode kernel).
static inline __m256i unpack_scales_mins_k4(const uint8_t *packed) {
  uint32_t utmp[4];
  memcpy(utmp, packed, K_SCALE_SIZE);
  utmp[3] =
      ((utmp[2] >> 4) & 0x0f0f0f0fu) | (((utmp[1] >> 6) & 0x03030303u) << 4);
  uint32_t uaux = utmp[1] & 0x3f3f3f3fu;
  utmp[1] = (utmp[2] & 0x0f0f0f0fu) | (((utmp[0] >> 6) & 0x03030303u) << 4);
  utmp[2] = uaux;
  utmp[0] &= 0x3f3f3f3fu;
  return _mm256_cvtepu8_epi16(_mm_set_epi32((int)utmp[3], (int)utmp[2],
                                            (int)utmp[1], (int)utmp[0]));
}

// Quantize one QK_K block of activations to int8 with a single scale (same
// idea as ggml's Q8_K). Returns the scale d so that x ~= d * q; 0 for an
// all-zero block.
static inline float quantize_block_q8(const float *x, int8_t *q) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  __m256 vmax = _mm256_setzero_ps();
  for (int i = 0; i < QK_K; i += 8)
    vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + i)));
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax),
                        _mm256_extractf128_ps(vmax, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_movehdup_ps(m));
  float amax = _mm_cvtss_f32(m);
  if (amax == 0.0f) {
    memset(q, 0, QK_K);
    return 0.0f;
  }

  float iscale = 127.0f / amax;
  const __m256 vscale = _mm256_set1_ps(iscale);
  const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (int i = 0; i < QK_K; i += 32) {
    __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i), vscale));
    __m256i i1 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vscale));
    __m256i i2 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vscale));
    __m256i i3 = _mm256_cvtps_epi32(
        _mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vscale));
    // the packs work per 128-bit lane, so fix the order up afterwards
    __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1),
                                        _mm256_packs_epi32(i2, i3));
    _mm256_storeu_si256((__m256i *)(q + i),
                        _mm256_permutevar8x32_epi32(packed, perm));
  }
  return 1.0f / iscale;
}

// Quantized activations for one matvec, so the dot with the quantized weights
// can stay in integers and handle 32 weights per instruction. bsums holds the
// sum of every 16 values; the kernels use it for the weight offsets (Q4_K mins,
// Q6_K's -32 bias).
static void quantize_x_q8(const float *x, int n, int8_t *q8, int16_t *bsums,
                          float *d8) {
  for (int b = 0; b < n / QK_K; b++) {
    int8_t *qb = q8 + b * QK_K;
    d8[b] = quantize_block_q8(x + b * QK_K, qb);
    for (int j = 0; j < 16; j++) {
      int sum = 0;
      for (int i = 0; i < 16; i++)
        sum += qb[j * 16 + i];
      bsums[b * 16 + j] = (int16_t)sum;
    }
  }
}

static inline __m256i hsum_i32_x4(__m256i a0, __m256i a1, __m256i a2,
                                  __m256i a3) {
  // -> 128-bit halves each holding [sum(a0), sum(a1), sum(a2), sum(a3)] parts
  return _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1),
                           _mm256_hadd_epi32(a2, a3));
}

// Horizontal sums of 8 int32 vectors, one result per lane.
static inline __m256i hsum_i32_x8(const __m256i *a) {
  __m256i lo = hsum_i32_x4(a[0], a[1], a[2], a[3]);
  __m256i hi = hsum_i32_x4(a[4], a[5], a[6], a[7]);
  __m128i s_lo = _mm_add_epi32(_mm256_castsi256_si128(lo),
                               _mm256_extracti128_si256(lo, 1));
  __m128i s_hi = _mm_add_epi32(_mm256_castsi256_si128(hi),
                               _mm256_extracti128_si256(hi, 1));
  return _mm256_set_m128i(s_hi, s_lo);
}

// ---- batched (prefill) path ------------------------------------------------
// Same layout idea as llama.cpp's tiled K-quant matmul: activations are
// quantized once per matmul and stored slab-major (all tokens' values for one
// QK_K-wide slab are contiguous), weights are unpacked one small window of rows
// at a time, and the kernel multiplies one weight row against 8 tokens at once.
// Everything the inner loop touches stays in L1.
#define TILE_ROWS 64
#define TILE_GROUP 8

// One unpacked Q4_K weight row (256 codes, 8 scales) against `width` tokens of
// one slab; returns the scaled integer dot per token (unused lanes are 0).
// width is always a constant at the call site, so the token loop unrolls and
// the accumulators stay in registers.
static inline __attribute__((always_inline)) __m256i
q4_group_dot(const uint8_t *w, const int16_t *scales, const int8_t *a,
             const int width) {
  __m256i sum[TILE_GROUP];
  for (int t = 0; t < TILE_GROUP; t++)
    sum[t] = _mm256_setzero_si256();
  for (int sb = 0; sb < 8; sb++) {
    __m256i wv = _mm256_load_si256((const __m256i *)(w + 32 * sb));
    __m256i scale = _mm256_set1_epi16(scales[sb]);
#pragma GCC unroll 8
    for (int t = 0; t < width; t++) {
      __m256i av =
          _mm256_load_si256((const __m256i *)(a + t * QK_K + 32 * sb));
      sum[t] = _mm256_add_epi32(
          sum[t], _mm256_madd_epi16(scale, _mm256_maddubs_epi16(wv, av)));
    }
  }
  return hsum_i32_x8(sum);
}

// Same for Q6_K: signed (code - 32) weights, 16 scales, activations biased to
// unsigned.
static inline __attribute__((always_inline)) __m256i
q6_group_dot(const int8_t *w, const int16_t *scales, const uint8_t *a,
             const int width) {
  __m256i sum[TILE_GROUP];
  for (int t = 0; t < TILE_GROUP; t++)
    sum[t] = _mm256_setzero_si256();
  for (int sb = 0; sb < 16; sb += 2) {
    __m256i wv = _mm256_load_si256((const __m256i *)(w + 16 * sb));
    __m256i scale = _mm256_set_m128i(_mm_set1_epi16(scales[sb + 1]),
                                     _mm_set1_epi16(scales[sb]));
#pragma GCC unroll 8
    for (int t = 0; t < width; t++) {
      __m256i av =
          _mm256_load_si256((const __m256i *)(a + t * QK_K + 16 * sb));
      sum[t] = _mm256_add_epi32(
          sum[t], _mm256_madd_epi16(scale, _mm256_maddubs_epi16(av, wv)));
    }
  }
  return hsum_i32_x8(sum);
}

typedef struct {
  int8_t *q;     // [slab][token][QK_K]; biased to unsigned (+128) for Q6_K
  int32_t *sums; // [slab][8][token], sum of each 32 values (Q4_K mins)
  float *d;      // [slab][token]
  int n_pad;     // tokens rounded up to TILE_GROUP
} BatchQ8;

// Reused across calls: prefill calls matmul 7 times per layer and these would
// otherwise be re-mapped and re-faulted every time. Not safe for concurrent
// matmul calls, which this program never makes.
static BatchQ8 batch_ws;
static size_t batch_ws_q_cap, batch_ws_sums_cap, batch_ws_d_cap;

static void *grow(void *p, size_t *cap, size_t need) {
  if (need <= *cap)
    return p;
  free(p);
  p = aligned_alloc(64, (need + 63) & ~(size_t)63);
  if (!p) {
    perror("aligned_alloc");
    abort();
  }
  *cap = need;
  return p;
}

static const BatchQ8 *quantize_batch(const float *X, int n, int in, int bias) {
  int nb = in / QK_K;
  int n_pad = (n + TILE_GROUP - 1) / TILE_GROUP * TILE_GROUP;
  BatchQ8 *ws = &batch_ws;
  ws->n_pad = n_pad;
  ws->q = grow(ws->q, &batch_ws_q_cap, (size_t)nb * n_pad * QK_K);
  ws->sums = grow(ws->sums, &batch_ws_sums_cap,
                  (size_t)nb * 8 * n_pad * sizeof(int32_t));
  ws->d = grow(ws->d, &batch_ws_d_cap, (size_t)nb * n_pad * sizeof(float));

#pragma omp parallel for
  for (int t = 0; t < n_pad; t++) {
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i flip = _mm256_set1_epi8((char)0x80);
    for (int b = 0; b < nb; b++) {
      int8_t *q = ws->q + ((size_t)b * n_pad + t) * QK_K;
      if (t >= n) { // padding tokens: scale 0, so they contribute nothing
        memset(q, 0, QK_K);
        ws->d[(size_t)b * n_pad + t] = 0.0f;
        for (int s = 0; s < 8; s++)
          ws->sums[((size_t)b * 8 + s) * n_pad + t] = 0;
        continue;
      }
      ws->d[(size_t)b * n_pad + t] =
          quantize_block_q8(X + (size_t)t * in + b * QK_K, q);
      for (int s = 0; s < 8; s++) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(q + 32 * s));
        __m256i sum = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, v), ones16);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                   _mm256_extracti128_si256(sum, 1));
        s4 = _mm_hadd_epi32(s4, s4);
        s4 = _mm_hadd_epi32(s4, s4);
        ws->sums[((size_t)b * 8 + s) * n_pad + t] = _mm_cvtsi128_si32(s4);
        if (bias)
          _mm256_storeu_si256((__m256i *)(q + 32 * s),
                              _mm256_xor_si256(v, flip));
      }
    }
  }
  return ws;
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

#ifdef __AVX2__
void matvec_q4_k(float *y, const BlockQ4_K *blocks, const float *x, int out,
                 int in) {
  int blocks_per_row = in / QK_K;
  int8_t q8[in];
  int16_t bsums[in / 16];
  float d8[blocks_per_row];
  quantize_x_q8(x, in, q8, bsums, d8);

#pragma omp parallel for
  for (int i = 0; i < out; i++) {
    const BlockQ4_K *row = blocks + (size_t)i * blocks_per_row;
    const __m256i mask15 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_min = _mm_setzero_ps();

    for (int b = 0; b < blocks_per_row; b++) {
      const BlockQ4_K *block = &row[b];
      float d = fp16_to_fp32(block->d) * d8[b];
      float dmin = -fp16_to_fp32(block->dmin) * d8[b];

      __m256i mins_and_scales = unpack_scales_mins_k4(block->scales);

      // min term: each min multiplies the sum of the 32 activations it covers
      __m256i q8sums = _mm256_loadu_si256((const __m256i *)(bsums + b * 16));
      __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                   _mm256_extracti128_si256(q8sums, 1));
      __m128i prod =
          _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
      acc_min =
          _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_min);

      __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
      __m256i scales = _mm256_set_m128i(sc128, sc128);
      __m256i sumi = _mm256_setzero_si256();
      const uint8_t *q4 = block->qs;
      const int8_t *xq = q8 + b * QK_K;

      for (int j = 0; j < QK_K / 64; j++) {
        // broadcast scale 2j (low nibbles) and 2j+1 (high nibbles) as int16
        __m256i scale_l = _mm256_shuffle_epi8(
            scales, _mm256_set1_epi16((short)(((4 * j + 1) << 8) | (4 * j))));
        __m256i scale_h = _mm256_shuffle_epi8(
            scales,
            _mm256_set1_epi16((short)(((4 * j + 3) << 8) | (4 * j + 2))));
        __m256i q4bits = _mm256_loadu_si256((const __m256i *)(q4 + 32 * j));
        __m256i q4l = _mm256_and_si256(q4bits, mask15);
        __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), mask15);
        __m256i p16l = _mm256_maddubs_epi16(
            q4l, _mm256_loadu_si256((const __m256i *)(xq + 64 * j)));
        __m256i p16h = _mm256_maddubs_epi16(
            q4h, _mm256_loadu_si256((const __m256i *)(xq + 64 * j + 32)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale_l, p16l));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale_h, p16h));
      }
      acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    y[i] = horizontal_sum(acc) + horizontal_sum128(acc_min);
  }
}
#else
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
#endif

// Batched over tokens: walk the weights once, dequantize a tile of rows into a
// float buffer and let sgemm multiply it with every token, instead of one full
// matvec (a full pass over the matrix) per token. 32 rows measured best: larger
// tiles fall out of cache for wide matrices when all threads are busy.
#define MATMUL_TILE_ROWS 32

#ifdef __AVX2__
void matmul_q4_k(float *Y, const BlockQ4_K *blocks, const float *X, int n,
                 int out, int in) {
  int nb = in / QK_K;
  const BatchQ8 *act = quantize_batch(X, n, in, 0);
  int n_pad = act->n_pad;

#pragma omp parallel for schedule(dynamic, 1)
  for (int i0 = 0; i0 < out; i0 += TILE_ROWS) {
    int rows = out - i0 < TILE_ROWS ? out - i0 : TILE_ROWS;
    // unpacked weights for this window and one slab: 4-bit codes as bytes in
    // natural order, plus per-32 scales and mins
    uint8_t codes[TILE_ROWS * QK_K] __attribute__((aligned(32)));
    int16_t scales[TILE_ROWS][8];
    int32_t mins[TILE_ROWS][8];
    float d[TILE_ROWS], dmin[TILE_ROWS];
    float acc[rows * n_pad];
    memset(acc, 0, sizeof(acc));
    const __m256i mask15 = _mm256_set1_epi8(0xF);

    for (int b = 0; b < nb; b++) {
      for (int r = 0; r < rows; r++) {
        const BlockQ4_K *block = blocks + (size_t)(i0 + r) * nb + b;
        d[r] = fp16_to_fp32(block->d);
        dmin[r] = fp16_to_fp32(block->dmin);
        __m256i mins_and_scales = unpack_scales_mins_k4(block->scales);
        _mm_storeu_si128((__m128i *)scales[r],
                         _mm256_castsi256_si128(mins_and_scales));
        _mm256_storeu_si256(
            (__m256i *)mins[r],
            _mm256_cvtepi16_epi32(
                _mm256_extracti128_si256(mins_and_scales, 1)));
        for (int j = 0; j < QK_K / 64; j++) {
          __m256i bits =
              _mm256_loadu_si256((const __m256i *)(block->qs + 32 * j));
          _mm256_store_si256((__m256i *)(codes + r * QK_K + 64 * j),
                             _mm256_and_si256(bits, mask15));
          _mm256_store_si256(
              (__m256i *)(codes + r * QK_K + 64 * j + 32),
              _mm256_and_si256(_mm256_srli_epi16(bits, 4), mask15));
        }
      }

      const int8_t *aq = act->q + (size_t)b * n_pad * QK_K;
      const int32_t *asums = act->sums + (size_t)b * 8 * n_pad;
      const float *ad = act->d + (size_t)b * n_pad;

      for (int t0 = 0; t0 < n_pad; t0 += TILE_GROUP) {
        const int8_t *a = aq + (size_t)t0 * QK_K;
        int wide = n - t0 > 4; // a short last group only needs 4 tokens' work
        for (int r = 0; r < rows; r++) {
          const uint8_t *w = codes + r * QK_K;
          __m256i total = wide ? q4_group_dot(w, scales[r], a, 8)
                               : q4_group_dot(w, scales[r], a, 4);

          // kept out of the dot above so its 8 accumulators stay in registers
          __m256i min_sum = _mm256_setzero_si256();
          for (int sb = 0; sb < 8; sb++)
            min_sum = _mm256_add_epi32(
                min_sum,
                _mm256_mullo_epi32(
                    _mm256_loadu_si256(
                        (const __m256i *)(asums + sb * n_pad + t0)),
                    _mm256_set1_epi32(mins[r][sb])));

          // weight = d * scale * code - dmin * min, for 8 tokens at once
          __m256 res =
              _mm256_mul_ps(_mm256_set1_ps(d[r]), _mm256_cvtepi32_ps(total));
          res = _mm256_fnmadd_ps(_mm256_set1_ps(dmin[r]),
                                 _mm256_cvtepi32_ps(min_sum), res);
          float *out_acc = acc + r * n_pad + t0;
          _mm256_storeu_ps(out_acc,
                           _mm256_fmadd_ps(res, _mm256_loadu_ps(ad + t0),
                                           _mm256_loadu_ps(out_acc)));
        }
      }
    }

    for (int r = 0; r < rows; r++)
      for (int t = 0; t < n; t++)
        Y[(size_t)t * out + i0 + r] = acc[r * n_pad + t];
  }
}
#else
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
#endif

#ifdef __AVX2__
void matvec_q6_k(float *y, const BlockQ6_K *blocks, const float *x, int out,
                 int in) {
  // entry g picks scale 2g for the low 8 bytes and scale 2g+1 for the high 8
  const __m128i q6_scale_shuffle[8] = {
      _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
      _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3),
      _mm_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5),
      _mm_setr_epi8(6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7),
      _mm_setr_epi8(8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9),
      _mm_setr_epi8(10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
                    11),
      _mm_setr_epi8(12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
                    13),
      _mm_setr_epi8(14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15,
                    15),
  };
  int blocks_per_row = in / QK_K;
  int8_t q8[in];
  int16_t bsums[in / 16];
  float d8[blocks_per_row];
  quantize_x_q8(x, in, q8, bsums, d8);

#pragma omp parallel for
  for (int i = 0; i < out; i++) {
    const BlockQ6_K *row = blocks + (size_t)i * blocks_per_row;
    const __m256i mask15 = _mm256_set1_epi8(0xF);
    const __m256i mask3 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < blocks_per_row; b++) {
      const BlockQ6_K *block = &row[b];
      float d = fp16_to_fp32(block->d) * d8[b];
      const int8_t *sc = block->scales;

      // weights are stored as q + 32, so subtract 32 * scale * sum(x) at the
      // end instead of un-biasing every weight
      __m256i q8sums = _mm256_loadu_si256((const __m256i *)(bsums + b * 16));
      __m128i scales8 = _mm_loadu_si128((const __m128i *)sc);
      __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
      __m256i bias =
          _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);
      __m256i sumi = _mm256_setzero_si256();
      const int8_t *xq = q8 + b * QK_K;

      for (int j = 0; j < QK_K / 128; j++) {
        __m256i hi =
            _mm256_loadu_si256((const __m256i *)(block->qh + 32 * j));
        __m256i lo1 =
            _mm256_loadu_si256((const __m256i *)(block->ql + 64 * j));
        __m256i lo2 =
            _mm256_loadu_si256((const __m256i *)(block->ql + 64 * j + 32));

        __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(hi, mask3), 4);
        __m256i h2 = _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(hi, 2), mask3), 4);
        __m256i h3 = _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(hi, 4), mask3), 4);
        __m256i h4 = _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(hi, 6), mask3), 4);
        __m256i q1 = _mm256_or_si256(_mm256_and_si256(lo1, mask15), h1);
        __m256i q2 = _mm256_or_si256(_mm256_and_si256(lo2, mask15), h2);
        __m256i q3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(lo1, 4), mask15), h3);
        __m256i q4 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(lo2, 4), mask15), h4);

        const int8_t *xj = xq + 128 * j;
        __m256i p1 = _mm256_maddubs_epi16(
            q1, _mm256_loadu_si256((const __m256i *)(xj)));
        __m256i p2 = _mm256_maddubs_epi16(
            q2, _mm256_loadu_si256((const __m256i *)(xj + 32)));
        __m256i p3 = _mm256_maddubs_epi16(
            q3, _mm256_loadu_si256((const __m256i *)(xj + 64)));
        __m256i p4 = _mm256_maddubs_epi16(
            q4, _mm256_loadu_si256((const __m256i *)(xj + 96)));

        // one scale per 16 weights: low lane and high lane of each product.
        // Shuffling them out of the already-loaded scale bytes is cheaper than
        // eight broadcasts from memory.
        __m256i s1 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales8, q6_scale_shuffle[4 * j + 0]));
        __m256i s2 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales8, q6_scale_shuffle[4 * j + 1]));
        __m256i s3 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales8, q6_scale_shuffle[4 * j + 2]));
        __m256i s4 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales8, q6_scale_shuffle[4 * j + 3]));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, p1));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, p2));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, p3));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, p4));
      }
      sumi = _mm256_sub_epi32(sumi, bias);
      acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    y[i] = horizontal_sum(acc);
  }
}
#else
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
#endif

#ifdef __AVX2__
void matmul_q6_k(float *Y, const BlockQ6_K *blocks, const float *X, int n,
                 int out, int in) {
  int nb = in / QK_K;
  // activations biased to unsigned so the signed (code - 32) weights can go
  // straight into maddubs; the bias is taken back out once per row below
  const BatchQ8 *act = quantize_batch(X, n, in, 1);
  int n_pad = act->n_pad;

#pragma omp parallel for schedule(dynamic, 1)
  for (int i0 = 0; i0 < out; i0 += TILE_ROWS) {
    int rows = out - i0 < TILE_ROWS ? out - i0 : TILE_ROWS;
    int8_t codes[TILE_ROWS * QK_K] __attribute__((aligned(32)));
    int16_t scales[TILE_ROWS][16];
    int32_t bias_fix[TILE_ROWS]; // 128 * sum(scale * code) for the row
    float d[TILE_ROWS];
    float acc[rows * n_pad];
    memset(acc, 0, sizeof(acc));
    const __m256i mask15 = _mm256_set1_epi8(0xF);
    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i off32 = _mm256_set1_epi8(32);
    const __m256i ones8 = _mm256_set1_epi8(1);

    for (int b = 0; b < nb; b++) {
      for (int r = 0; r < rows; r++) {
        const BlockQ6_K *block = blocks + (size_t)(i0 + r) * nb + b;
        d[r] = fp16_to_fp32(block->d);
        for (int j = 0; j < 16; j++)
          scales[r][j] = block->scales[j];

        __m256i fix = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 128; j++) {
          __m256i hi =
              _mm256_loadu_si256((const __m256i *)(block->qh + 32 * j));
          __m256i lo1 =
              _mm256_loadu_si256((const __m256i *)(block->ql + 64 * j));
          __m256i lo2 =
              _mm256_loadu_si256((const __m256i *)(block->ql + 64 * j + 32));
          __m256i q[4];
          q[0] = _mm256_or_si256(
              _mm256_and_si256(lo1, mask15),
              _mm256_slli_epi16(_mm256_and_si256(hi, mask3), 4));
          q[1] = _mm256_or_si256(
              _mm256_and_si256(lo2, mask15),
              _mm256_slli_epi16(
                  _mm256_and_si256(_mm256_srli_epi16(hi, 2), mask3), 4));
          q[2] = _mm256_or_si256(
              _mm256_and_si256(_mm256_srli_epi16(lo1, 4), mask15),
              _mm256_slli_epi16(
                  _mm256_and_si256(_mm256_srli_epi16(hi, 4), mask3), 4));
          q[3] = _mm256_or_si256(
              _mm256_and_si256(_mm256_srli_epi16(lo2, 4), mask15),
              _mm256_slli_epi16(
                  _mm256_and_si256(_mm256_srli_epi16(hi, 6), mask3), 4));
          for (int k = 0; k < 4; k++) {
            int sb = 8 * j + 2 * k; // two 16-wide scales per 32 codes
            __m256i wv = _mm256_sub_epi8(q[k], off32);
            _mm256_store_si256(
                (__m256i *)(codes + r * QK_K + 128 * j + 32 * k), wv);
            __m256i scale =
                _mm256_set_m128i(_mm_set1_epi16(scales[r][sb + 1]),
                                 _mm_set1_epi16(scales[r][sb]));
            fix = _mm256_add_epi32(
                fix,
                _mm256_madd_epi16(scale, _mm256_maddubs_epi16(ones8, wv)));
          }
        }
        __m128i f4 = _mm_add_epi32(_mm256_castsi256_si128(fix),
                                   _mm256_extracti128_si256(fix, 1));
        f4 = _mm_hadd_epi32(f4, f4);
        f4 = _mm_hadd_epi32(f4, f4);
        bias_fix[r] = 128 * _mm_cvtsi128_si32(f4);
      }

      const uint8_t *aq = (const uint8_t *)act->q + (size_t)b * n_pad * QK_K;
      const float *ad = act->d + (size_t)b * n_pad;

      for (int t0 = 0; t0 < n_pad; t0 += TILE_GROUP) {
        const uint8_t *a = aq + (size_t)t0 * QK_K;
        int wide = n - t0 > 4;
        for (int r = 0; r < rows; r++) {
          const int8_t *w = codes + r * QK_K;
          __m256i dots = wide ? q6_group_dot(w, scales[r], a, 8)
                              : q6_group_dot(w, scales[r], a, 4);
          __m256i total =
              _mm256_sub_epi32(dots, _mm256_set1_epi32(bias_fix[r]));
          __m256 res =
              _mm256_mul_ps(_mm256_set1_ps(d[r]), _mm256_cvtepi32_ps(total));
          float *out_acc = acc + r * n_pad + t0;
          _mm256_storeu_ps(out_acc,
                           _mm256_fmadd_ps(res, _mm256_loadu_ps(ad + t0),
                                           _mm256_loadu_ps(out_acc)));
        }
      }
    }

    for (int r = 0; r < rows; r++)
      for (int t = 0; t < n; t++)
        Y[(size_t)t * out + i0 + r] = acc[r * n_pad + t];
  }
}
#else
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
#endif
