#include "ops.h"
#include <cmath>
#include <math.h>
#include <stdint.h>
#include <string.h>

float bf16_to_float32(uint16_t in) {
  // cool trick, you cast it to 32 bit to get
  // 0...0 (16bits) 1..0 (existing bits from in)
  // then when you do << 16 we add 16 0s at the right so we get
  // 1..0(existing bits from in) 0..0 (16 bits of 0) => this is the same format
  // for float!!!! where first bit is signed (same for BF16), first 8 bits are
  // the scaling bits (same for BF16) and finally rest is all mantissa which is
  // also the same for BF16 (trailing 0s don't make a difference)
  uint32_t t = ((uint32_t)in) << 16;
  float out;
  memcpy(&out, &t, sizeof(out));
  return out;
}

void lookup(float *x, const uint16_t *token_emb, int token_id, int hidden) {
  for (int i = 0; i < hidden; i++) {
    x[i] = bf16_to_float32(token_emb[token_id * hidden + i]);
  }
}

void rmsnorm(float *xn, const float *x, const uint16_t *weight, int n,
             float eps) {
  // calculate RMS => then divide by it to scale the number (multiply with the
  // weights)
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }

  float rms = sqrtf(sum / (float)n + eps);
  for (int i = 0; i < n; i++) {
    xn[i] = (x[i] / rms) * bf16_to_float32(weight[i]);
  }
}

void matvec(float *y, const uint16_t *W, const float *x, int out, int in) {
  for (int i = 0; i < out; i++) {
    float sum = 0.0f;
    for (int j = 0; j < in; j++) {
      sum += bf16_to_float32(W[i * in + j]) * x[j];
    }
    y[i] = sum;
  }
}

void add(float *x, const float *branch, int n) {
  for (int i = 0; i < n; i++) {
    x[i] += branch[i];
  }
}

void silu(float *x, int n) {
  for (int i = 0; i < n; i++) {
    float z = x[i];
    x[i] = z / (1 + expf(z));
  }
}

void softmax(float *x, int n) {
  float max = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max)
      max = x[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max);
    sum += x[i];
  }

  for (int i = 0; i < n; i++) {
    x[i] /= sum;
  }
}
