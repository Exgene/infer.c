#include "ops.h"
#include "config.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
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

void attention(float *out, const float *q, const float *k, const float *v,
               int seq_len, WeightsConfigJson *cfg) {
  // to avoid pointer inderection store them as ints
  int head_dim = cfg->head_dim;
  int num_kv_heads = cfg->num_kv_heads;
  int num_heads = cfg->num_heads;

  // derive things like scale factor, kv_dim etc.
  int kv_dim = num_kv_heads * head_dim;
  float scale = 1.0f / sqrtf((float)head_dim);
  float *score = malloc(sizeof(float) * seq_len);

  // For each head we calculate the attention!
  for (int h = 0; h < num_heads; h++) {
    int kv_h = h / (num_heads / num_kv_heads);
    const float *qh = q + h * head_dim;

    // for each token we need to calculate its key value and dot product it with
    // query.
    for (int t = 0; t < seq_len; t++) {
      const float *kh = k + kv_h * head_dim + t * kv_dim;
      float dot = 0.0f;
      // And we are offsetting by head_dim (i think its 64 in this case)
      for (int d = 0; d < head_dim; d++) {
        dot += qh[d] * kh[d];
      }
      score[t] = dot * scale;
    }

    softmax(score, seq_len);

    float *o = out + head_dim * h;
    for (int i = 0; i < head_dim; i++) {
      o[i] = 0.0f;
    }

    // same as key we are doign it with value, and multiply by score to get the
    // attention output.
    for (int t = 0; t < seq_len; t++) {
      const float *vh = v + kv_h * head_dim + t * kv_dim;
      for (int d = 0; d < head_dim; d++) {
        out[d] += vh[d] * score[t];
      }
    }
  }
  free(score);
}
