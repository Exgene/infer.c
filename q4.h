#ifndef Q4_H
#define Q4_H

#include "weight.h"

static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = h & 0x03FFu;
  uint32_t bits;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 14;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }

  float out;
  __builtin_memcpy(&out, &bits, sizeof(out));
  return out;
}

void dequantize_row_q4_k(const BlockQ4_K *blocks, float *y, int k);
void dequantize_row_q6_k(const BlockQ6_K *blocks, float *y, int k);
void matvec_q4_k(float *y, const BlockQ4_K *blocks, const float *x, int out,
                 int in);
void matmul_q4_k(float *Y, const BlockQ4_K *blocks, const float *X, int n,
                 int out, int in);
void matvec_q6_k(float *y, const BlockQ6_K *blocks, const float *x, int out,
                 int in);
void matmul_q6_k(float *Y, const BlockQ6_K *blocks, const float *X, int n,
                 int out, int in);

#endif
