#ifndef OPS_H
#define OPS_H

#include "config.h"
#include "weights.h"
#include <stdint.h>
#include <string.h>

typedef struct {
  float p;
  int id;
} Prob;

static inline float bf16_to_float32(uint16_t in) {
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

void lookup(float *x, const Weight *token_emb, int token_id);

void rmsnorm(float *xn, const float *x, float *weight, int n, float eps);

void matvec(float *y, const Weight *W, const float *x);

void add(float *x, const float *branch, int n);

void silu(float *x, int n);

void attention(float *out, const float *q, const float *k, const float *v,
               int seq_len, int head_dim, int num_kv_heads, int num_heads,
               float *score);

int forward(const WeightsConfigJson *cfg, const Weights *w, float *x, float *xn,
            float *q, float *k, float *v, float *attn, float *hb, float *hb2,
            float *logits, int token_id, int pos, float *k_cache,
            float *v_cache, int max_seq, float scale, float low, float high,
            int orig, float *score);

void rope_rotation(float *x, int position, int head_dim, float rope_theta,
                   float scale, float low, float high, int orig);

int sample_top_p(float *logits, int vocab, float p, float temp, Prob *ps);

void matmul(float *Y, const Weight *W, const float *X, int n);

void lookup_batch(float *X, const Weight *token_emb, const int *tokens, int n);

void rmsnorm_batch(float *Xn, const float *X, float *weight, int n_rows,
                   int hidden, float eps);

void add_batch(float *X, const float *branch, int n_rows, int len);

void silu_batch(float *X, int n_rows, int len);

int forward_prefill(const WeightsConfigJson *cfg, const Weights *w, float *x,
                     float *xn, float *q, float *k, float *v, float *attn,
                     float *hb, float *hb2, float *logits, const int *tokens,
                     int n, float *k_cache, float *v_cache, int max_seq,
                     float scale, float low, float high, int orig,
                     float *score);
#endif
