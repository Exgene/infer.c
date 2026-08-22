#include "ops.h"
#include "config.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static float llama3_freq(float freq, float scale, float low, float high,
                         int orig) {
  float wavelen = 6.283185307f / freq;
  float low_w = (float)orig / low;
  float high_w = (float)orig / high;
  if (wavelen < high_w)
    return freq;
  if (wavelen > low_w)
    return freq / scale;
  float smooth = ((float)orig / wavelen - low) / (high - low);
  return (1.0f - smooth) * (freq / scale) + smooth * freq;
}

void rope_rotation(float *x, int position, int head_dim, float rope_theta,
                   float scale, float low, float high, int orig) {
  int half = head_dim / 2;
  for (int i = 0; i < half; i++) {
    float freq = 1.0f / powf(rope_theta, (2.0f * (float)i) / (float)head_dim);
    freq = llama3_freq(freq, scale, low, high, orig);
    float theta = (float)position * freq;

    float c = cosf(theta);
    float s = sinf(theta);

    float x0 = x[i];
    float x1 = x[i + half];

    x[i] = x0 * c - x1 * s;
    x[i + half] = x0 * s + x1 * c;
  }
}

void lookup(float *x, float *token_emb, int token_id, int hidden) {
  for (int i = 0; i < hidden; i++) {
    x[i] = token_emb[token_id * hidden + i];
  }
}

void rmsnorm(float *xn, const float *x, float *weight, int n, float eps) {
  // calculate RMS => then divide by it to scale the number (multiply with the
  // weights)
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }

  float rms = sqrtf(sum / (float)n + eps);
  for (int i = 0; i < n; i++) {
    xn[i] = (x[i] / rms) * weight[i];
  }
}

void matvec(float *y, float *W, const float *x, int out, int in) {
#pragma omp parallel for
  for (int i = 0; i < out; i++) {
    float sum = 0.0f;
    for (int j = 0; j < in; j++) {
      sum += W[i * in + j] * x[j];
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
    x[i] = z / (1 + expf(-z));
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
               int seq_len, int head_dim, int num_kv_heads, int num_heads,
               float *score) {
  // derive things like scale factor, kv_dim etc.
  int kv_dim = num_kv_heads * head_dim;
  float scale = 1.0f / sqrtf((float)head_dim);
  memset(score, 0, seq_len * sizeof(float));

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
        o[d] += vh[d] * score[t];
      }
    }
  }
}

static float *cache_slot(float *cache, int layer, int pos, int max_seq,
                         int kv_dim) {
  return cache + ((size_t)layer * max_seq + pos) * kv_dim;
}

static float *cache_layer(float *cache, int layer, int max_seq, int kv_dim) {
  return cache + (size_t)layer * max_seq * kv_dim;
}

int forward(const WeightsConfigJson *cfg, const Weights *w, float *x, float *xn,
            float *q, float *k, float *v, float *attn, float *hb, float *hb2,
            float *logits, int token_id, int pos, float *k_cache,
            float *v_cache, int max_seq, float scale, float low, float high,
            int orig, float *score) {
  // to avoid pointer inderection take once and use multiple times!
  int hidden = cfg->hidden_size;
  int num_layers = cfg->num_layers;
  int head_dim = cfg->head_dim;
  int num_kv_heads = cfg->num_kv_heads;
  int kv_dim = head_dim * num_kv_heads;
  float eps = cfg->rms_norm_eps;
  int num_heads = cfg->num_heads;
  int intermediate = cfg->intermediate_size;
  int vocab = cfg->vocab_size;
  float rope_theta = cfg->rope_theta;

  lookup(x, w->token_emb, token_id, hidden);

  for (int l = 0; l < num_layers; l++) {
    const Layer *layer = &w->layers[l];
    rmsnorm(xn, x, layer->rms_att, hidden, eps);

    // for attention mechanism calculate q @ Wq, k @ Wk, v @ Wv
    matvec(q, layer->wq, xn, hidden, hidden);
    matvec(k, layer->wk, xn, kv_dim, hidden);
    matvec(v, layer->wv, xn, kv_dim, hidden);

    for (int h = 0; h < num_heads; h++)
      rope_rotation(q + h * head_dim, pos, head_dim, rope_theta, scale, low,
                    high, orig);

    for (int h = 0; h < num_kv_heads; h++)
      rope_rotation(k + h * head_dim, pos, head_dim, rope_theta, scale, low,
                    high, orig);

    memcpy(cache_slot(k_cache, l, pos, max_seq, kv_dim), k,
           kv_dim * sizeof(float));
    memcpy(cache_slot(v_cache, l, pos, max_seq, kv_dim), v,
           kv_dim * sizeof(float));

    float *k_base = cache_layer(k_cache, l, max_seq, kv_dim);
    float *v_base = cache_layer(v_cache, l, max_seq, kv_dim);

    // do the Grouped Query Attention for this pass.
    attention(attn, q, k_base, v_base, pos + 1, head_dim, num_kv_heads,
              num_heads, score);
    matvec(xn, layer->wo, attn, hidden, hidden);
    add(x, xn, hidden);

    // MLP forward pass
    rmsnorm(xn, x, layer->rms_ffn, hidden, eps);
    matvec(hb, layer->w_gate, xn, intermediate, hidden);
    matvec(hb2, layer->w_up, xn, intermediate, hidden);
    silu(hb, intermediate);
    for (int i = 0; i < intermediate; i++) {
      hb[i] *= hb2[i];
    }
    matvec(xn, layer->w_down, hb, hidden, intermediate);
    add(x, xn, hidden);
  }

  rmsnorm(xn, x, w->rms_final, hidden, eps);
  matvec(logits, w->token_emb, xn, vocab, hidden);
  int best = 0;
  for (int i = 1; i < vocab; i++) {
    if (logits[i] > logits[best])
      best = i;
  }
  return best;
}

static int cmp_prob_desc(const void *a, const void *b) {
  float da = ((const Prob *)a)->p;
  float db = ((const Prob *)b)->p;
  if (da < db)
    return 1;
  if (da > db)
    return -1;
  return 0;
}

int sample_top_p(float *logits, int vocab, float p, float temp, Prob *ps) {
  if (temp < 1e-6f)
    temp = 1.0f;
  for (int i = 0; i < vocab; i++)
    logits[i] /= temp;
  softmax(logits, vocab);

  for (int i = 0; i < vocab; i++) {
    ps[i].p = logits[i];
    ps[i].id = i;
  }
  qsort(ps, (size_t)vocab, sizeof(Prob), cmp_prob_desc);

  float cum = 0.0f;
  int last = 0;
  for (int i = 0; i < vocab; i++) {
    cum += ps[i].p;
    last = i;
    if (cum >= p)
      break;
  }

  float u = ((float)rand() / (float)RAND_MAX) * cum;
  float run = 0.0f;
  int id = ps[last].id;
  for (int i = 0; i <= last; i++) {
    run += ps[i].p;
    if (u <= run) {
      id = ps[i].id;
      break;
    }
  }
  return id;
}
