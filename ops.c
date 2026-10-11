#include "ops.h"
#include "config.h"
#include "q4.h"
#include <cblas.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>

// exp() for 8 floats: range-reduce by powers of two, then a degree-5
// polynomial (the Cephes expf coefficients). Accurate to ~1e-7 relative; the
// input is clamped so the 2^n step can never leave float range.
static inline __m256 exp256_ps(__m256 x) {
  x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
  __m256 fx = _mm256_round_ps(
      _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 x2 = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_add_ps(_mm256_fmadd_ps(y, x2, x), _mm256_set1_ps(1.0f));
  __m256i pow2 = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2));
}
#endif

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

// cos/sin for every frequency at one position, as (cos, sin) pairs. The angles
// only depend on the position, so compute them once per token instead of once
// per head per layer.
static void rope_table(float *cs, int position, int head_dim, float rope_theta,
                       float scale, float low, float high, int orig) {
  int half = head_dim / 2;
  for (int i = 0; i < half; i++) {
    float freq = 1.0f / powf(rope_theta, (2.0f * (float)i) / (float)head_dim);
    freq = llama3_freq(freq, scale, low, high, orig);
    float theta = (float)position * freq;
    cs[2 * i] = cosf(theta);
    cs[2 * i + 1] = sinf(theta);
  }
}

static void apply_rope_table(float *x, const float *cs, int head_dim,
                             int interleaved) {
  int half = head_dim / 2;
  for (int i = 0; i < half; i++) {
    float c = cs[2 * i];
    float s = cs[2 * i + 1];
    float *a = interleaved ? &x[2 * i] : &x[i];
    float *b = interleaved ? &x[2 * i + 1] : &x[i + half];
    float x0 = *a;
    float x1 = *b;
    *a = x0 * c - x1 * s;
    *b = x0 * s + x1 * c;
  }
}

void lookup(float *x, const Weight *token_emb, int token_id) {
  weight_lookup_row(x, token_emb, token_id);
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

static void matvec_f32(float *y, const float *W, const float *x, int out,
                     int in) {
#pragma omp parallel for
  for (int i = 0; i < out; i++) {
    float sum = 0.0f;
    for (int j = 0; j < in; j++) {
      sum += W[(size_t)i * in + j] * x[j];
    }
    y[i] = sum;
  }
}

void matvec(float *y, const Weight *W, const float *x) {
  switch (W->type) {
  case WT_F32:
    matvec_f32(y, W->data.f32, x, W->rows, W->cols);
    break;
  case WT_Q4_K:
    matvec_q4_k(y, W->data.q4_k, x, W->rows, W->cols);
    break;
  case WT_Q6_K:
    matvec_q6_k(y, W->data.q6_k, x, W->rows, W->cols);
    break;
  default:
    break;
  }
}

void add(float *x, const float *branch, int n) {
  for (int i = 0; i < n; i++) {
    x[i] += branch[i];
  }
}

void silu(float *x, int n) {
  int i = 0;
#ifdef __AVX2__
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 sign = _mm256_set1_ps(-0.0f);
  for (; i + 8 <= n; i += 8) {
    __m256 z = _mm256_loadu_ps(x + i);
    __m256 e = exp256_ps(_mm256_xor_ps(z, sign));
    _mm256_storeu_ps(x + i, _mm256_div_ps(z, _mm256_add_ps(one, e)));
  }
#endif
  for (; i < n; i++) {
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
  int i = 0;
#ifdef __AVX2__
  const __m256 vmax = _mm256_set1_ps(max);
  __m256 vsum = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmax));
    _mm256_storeu_ps(x + i, e);
    vsum = _mm256_add_ps(vsum, e);
  }
  __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(vsum),
                         _mm256_extractf128_ps(vsum, 1));
  s4 = _mm_hadd_ps(s4, s4);
  s4 = _mm_hadd_ps(s4, s4);
  sum = _mm_cvtss_f32(s4);
#endif
  for (; i < n; i++) {
    x[i] = expf(x[i] - max);
    sum += x[i];
  }

  for (int j = 0; j < n; j++) {
    x[j] /= sum;
  }
}

void attention(float *out, const float *q, const float *k, const float *v,
               int seq_len, int head_dim, int num_kv_heads, int num_heads,
               float *score) {
  // derive things like scale factor, kv_dim etc.
  int kv_dim = num_kv_heads * head_dim;
  float scale = 1.0f / sqrtf((float)head_dim);
  (void)score; // each head keeps its own scores now that heads run in parallel

  // For each head we calculate the attention! Heads are independent, so share
  // them across threads once there is enough context to be worth it.
#pragma omp parallel for if (seq_len >= 64)
  for (int h = 0; h < num_heads; h++) {
    float score[seq_len];
    int kv_h = h / (num_heads / num_kv_heads);
    const float *qh = q + h * head_dim;

    // for each token we need to calculate its key value and dot product it with
    // query.
    for (int t = 0; t < seq_len; t++) {
      const float *kh = k + kv_h * head_dim + t * kv_dim;
      float dot = 0.0f;
      // And we are offsetting by head_dim (i think its 64 in this case)
#pragma omp simd reduction(+ : dot)
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

// Attention for n new tokens at positions pos0..pos0+n-1 against the cache, as
// two matrix products per head: scores = Q K^T, then out = softmax(scores) V.
// Token t may only see positions 0..pos0+t, so the rest of its row is zeroed.
static void attention_batch(float *out, const float *q, const float *k,
                            const float *v, int n, int pos0, int hidden,
                            int head_dim, int num_kv_heads, int num_heads) {
  int kv_dim = num_kv_heads * head_dim;
  int ctx = pos0 + n;
  float scale = 1.0f / sqrtf((float)head_dim);

#pragma omp parallel
  {
    float *scores = malloc((size_t)n * ctx * sizeof(float));
    if (!scores) {
      perror("malloc");
      abort();
    }

#pragma omp for schedule(dynamic, 1)
    for (int h = 0; h < num_heads; h++) {
      int kv_h = h / (num_heads / num_kv_heads);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, ctx, head_dim,
                  scale, q + h * head_dim, hidden, k + kv_h * head_dim, kv_dim,
                  0.0f, scores, ctx);
      for (int t = 0; t < n; t++) {
        int seq_len = pos0 + t + 1;
        softmax(scores + (size_t)t * ctx, seq_len);
        memset(scores + (size_t)t * ctx + seq_len, 0,
               (size_t)(ctx - seq_len) * sizeof(float));
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, head_dim, ctx,
                  1.0f, scores, ctx, v + kv_h * head_dim, kv_dim, 0.0f,
                  out + h * head_dim, hidden);
    }

    free(scores);
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

  lookup(x, &w->token_emb, token_id);

  float rope_cs[head_dim];
  rope_table(rope_cs, pos, head_dim, rope_theta, scale, low, high, orig);

  for (int l = 0; l < num_layers; l++) {
    const Layer *layer = &w->layers[l];
    rmsnorm(xn, x, layer->rms_att, hidden, eps);

    // for attention mechanism calculate q @ Wq, k @ Wk, v @ Wv
    matvec(q, &layer->wq, xn);
    matvec(k, &layer->wk, xn);
    matvec(v, &layer->wv, xn);

    for (int h = 0; h < num_heads; h++)
      apply_rope_table(q + h * head_dim, rope_cs, head_dim, w->qk_interleaved);

    for (int h = 0; h < num_kv_heads; h++)
      apply_rope_table(k + h * head_dim, rope_cs, head_dim, w->qk_interleaved);

    memcpy(cache_slot(k_cache, l, pos, max_seq, kv_dim), k,
           kv_dim * sizeof(float));
    memcpy(cache_slot(v_cache, l, pos, max_seq, kv_dim), v,
           kv_dim * sizeof(float));

    float *k_base = cache_layer(k_cache, l, max_seq, kv_dim);
    float *v_base = cache_layer(v_cache, l, max_seq, kv_dim);

    // do the Grouped Query Attention for this pass.
    attention(attn, q, k_base, v_base, pos + 1, head_dim, num_kv_heads,
              num_heads, score);
    matvec(xn, &layer->wo, attn);
    add(x, xn, hidden);

    // MLP forward pass
    rmsnorm(xn, x, layer->rms_ffn, hidden, eps);
    matvec(hb, &layer->w_gate, xn);
    matvec(hb2, &layer->w_up, xn);
    silu(hb, intermediate);
    for (int i = 0; i < intermediate; i++) {
      hb[i] *= hb2[i];
    }
    matvec(xn, &layer->w_down, hb);
    add(x, xn, hidden);
  }

  rmsnorm(xn, x, w->rms_final, hidden, eps);
  matvec(logits, &w->token_emb, xn);
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

void matmul(float *Y, const Weight *W, const float *X, int n) {
  if (n <= 0 || W->rows <= 0 || W->cols <= 0)
    return;
  if (W->type == WT_F32) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, W->rows, W->cols,
                1.0f, X, W->cols, W->data.f32, W->cols, 0.0f, Y, W->rows);
    return;
  }
  if (W->type == WT_Q4_K)
    matmul_q4_k(Y, W->data.q4_k, X, n, W->rows, W->cols);
  if (W->type == WT_Q6_K)
    matmul_q6_k(Y, W->data.q6_k, X, n, W->rows, W->cols);
}

void lookup_batch(float *X, const Weight *token_emb, const int *tokens, int n) {
  int hidden = token_emb->cols;
  for (int t = 0; t < n; t++) {
    weight_lookup_row(X + (size_t)t * hidden, token_emb, tokens[t]);
  }
}

void rmsnorm_batch(float *Xn, const float *X, float *weight, int n_rows,
                   int hidden, float eps) {
#pragma omp parallel for
  for (int t = 0; t < n_rows; t++) {
    rmsnorm(Xn + (size_t)t * hidden, X + (size_t)t * hidden, weight, hidden,
            eps);
  }
}

void add_batch(float *X, const float *branch, int n_rows, int len) {
  for (int t = 0; t < n_rows; t++) {
    add(X + (size_t)t * len, branch + (size_t)t * len, len);
  }
}

void silu_batch(float *X, int n_rows, int len) {
#pragma omp parallel for
  for (int t = 0; t < n_rows; t++) {
    silu(X + len * (size_t)t, len);
  }
}

// Call it for prefill only, batched operation over tokens. Handles the n tokens
// at positions pos0..pos0+n-1; earlier positions must already be in the cache,
// so a long prompt can be fed in chunks with buffers sized for one chunk. Only
// the last chunk needs logits; pass need_logits = 0 for the others (returns -1).
int forward_prefill(const WeightsConfigJson *cfg, const Weights *w, float *x,
                    float *xn, float *q, float *k, float *v, float *attn,
                    float *hb, float *hb2, float *logits, const int *tokens,
                    int n, int pos0, int need_logits, float *k_cache,
                    float *v_cache, int max_seq, float scale, float low,
                    float high, int orig, float *score) {
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

  (void)score;
  lookup_batch(x, &w->token_emb, tokens, n);

  float *rope_cs = malloc((size_t)n * head_dim * sizeof(float));
  if (!rope_cs)
    return -1;
  for (int t = 0; t < n; t++)
    rope_table(rope_cs + (size_t)t * head_dim, pos0 + t, head_dim, rope_theta,
               scale, low, high, orig);

  for (int l = 0; l < num_layers; l++) {
    const Layer *layer = &w->layers[l];
    rmsnorm_batch(xn, x, layer->rms_att, n, hidden, eps);

    // for attention mechanism calculate q @ Wq, k @ Wk, v @ Wv
    matmul(q, &layer->wq, xn, n);
    matmul(k, &layer->wk, xn, n);
    matmul(v, &layer->wv, xn, n);

#pragma omp parallel for
    for (int t = 0; t < n; t++) {
      const float *cs = rope_cs + (size_t)t * head_dim;
      for (int h = 0; h < num_heads; h++)
        apply_rope_table(q + (size_t)t * hidden + h * head_dim, cs, head_dim,
                         w->qk_interleaved);

      for (int h = 0; h < num_kv_heads; h++)
        apply_rope_table(k + (size_t)t * kv_dim + h * head_dim, cs, head_dim,
                         w->qk_interleaved);

      memcpy(cache_slot(k_cache, l, pos0 + t, max_seq, kv_dim),
             k + (size_t)t * kv_dim,
             kv_dim * sizeof(float));
      memcpy(cache_slot(v_cache, l, pos0 + t, max_seq, kv_dim),
             v + (size_t)t * kv_dim,
             kv_dim * sizeof(float));
    }

    float *k_base = cache_layer(k_cache, l, max_seq, kv_dim);
    float *v_base = cache_layer(v_cache, l, max_seq, kv_dim);

    // do the Grouped Query Attention for this pass.
    attention_batch(attn, q, k_base, v_base, n, pos0, hidden, head_dim,
                    num_kv_heads, num_heads);
    matmul(xn, &layer->wo, attn, n);
    add_batch(x, xn, n, hidden);

    // MLP forward pass
    rmsnorm_batch(xn, x, layer->rms_ffn, n, hidden, eps);
    matmul(hb, &layer->w_gate, xn, n);
    matmul(hb2, &layer->w_up, xn, n);
    silu_batch(hb, n, intermediate);
#pragma omp parallel for
    for (int t = 0; t < n; t++) {
      float *hb_base = hb + (size_t)t * intermediate;
      float *hb2_base = hb2 + (size_t)t * intermediate;
      for (int i = 0; i < intermediate; i++) {
        hb_base[i] *= hb2_base[i];
      }
    }
    matmul(xn, &layer->w_down, hb, n);
    add_batch(x, xn, n, hidden);
  }

  free(rope_cs);
  if (!need_logits)
    return -1;

  rmsnorm(xn + (size_t)(n - 1) * hidden, x + (size_t)(n - 1) * hidden,
          w->rms_final, hidden, eps);
  matvec(logits, &w->token_emb, xn + (size_t)(n - 1) * hidden);
  int best = 0;
  for (int i = 1; i < vocab; i++) {
    if (logits[i] > logits[best])
      best = i;
  }
  return best;
}
