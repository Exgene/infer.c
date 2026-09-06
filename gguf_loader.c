#include "gguf_loader.h"

#include "gguf.h"
#include "q4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tensor_rows_cols(const GgufTensor *t, int *rows, int *cols) {
  if (t->ndim == 1) {
    *rows = 1;
    *cols = (int)t->dims[0];
    return 0;
  }
  if (t->ndim == 2) {
    *cols = (int)t->dims[0];
    *rows = (int)t->dims[1];
    return 0;
  }
  return -1;
}

static float *convert_f16_vector(const uint16_t *src, size_t n) {
  float *dst = malloc(n * sizeof(float));
  if (!dst)
    return NULL;
  for (size_t i = 0; i < n; i++)
    dst[i] = fp16_to_fp32(src[i]);
  return dst;
}

static Weight convert_f16_matrix(const uint16_t *src, int rows, int cols) {
  size_t n = (size_t)rows * cols;
  float *dst = malloc(n * sizeof(float));
  if (!dst)
    return (Weight){0};
  for (size_t i = 0; i < n; i++)
    dst[i] = fp16_to_fp32(src[i]);
  return weight_make_f32(dst, rows, cols, 1);
}

static Weight bind_gguf_matrix(GgufFile *gf, const GgufTensor *t) {
  int rows = 0;
  int cols = 0;
  if (tensor_rows_cols(t, &rows, &cols) != 0)
    return (Weight){0};

  const void *ptr = gguf_tensor_data(gf, t);
  if (!ptr)
    return (Weight){0};

  switch (t->type) {
  case GGML_TYPE_F32:
    return weight_make_f32((float *)ptr, rows, cols, 0);
  case GGML_TYPE_F16:
    return convert_f16_matrix((const uint16_t *)ptr, rows, cols);
  case GGML_TYPE_Q4_K:
    return weight_make_q4_k((const BlockQ4_K *)ptr, rows, cols);
  case GGML_TYPE_Q6_K:
    return weight_make_q6_k((const BlockQ6_K *)ptr, rows, cols);
  default:
    fprintf(stderr, "gguf: unsupported matrix type %u for %s\n", t->type,
            t->name);
    return (Weight){0};
  }
}

static float *bind_gguf_norm(GgufFile *gf, const GgufTensor *t) {
  const void *ptr = gguf_tensor_data(gf, t);
  if (!ptr)
    return NULL;

  size_t n = 1;
  for (int i = 0; i < t->ndim; i++)
    n *= (size_t)t->dims[i];

  switch (t->type) {
  case GGML_TYPE_F32: {
    float *dst = malloc(n * sizeof(float));
    if (!dst)
      return NULL;
    memcpy(dst, ptr, n * sizeof(float));
    return dst;
  }
  case GGML_TYPE_F16:
    return convert_f16_vector((const uint16_t *)ptr, n);
  default:
    fprintf(stderr, "gguf: unsupported norm type %u for %s\n", t->type,
            t->name);
    return NULL;
  }
}

static int bind_gguf_layer(Weights *w, GgufFile *gf, int layer) {
  char name[256];
  Layer *L = &w->layers[layer];
  const GgufTensor *t = NULL;

  snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->rms_att = bind_gguf_norm(gf, t);
  if (!L->rms_att)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->rms_ffn = bind_gguf_norm(gf, t);
  if (!L->rms_ffn)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->wq = bind_gguf_matrix(gf, t);
  if (L->wq.type == WT_NONE)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->wk = bind_gguf_matrix(gf, t);
  if (L->wk.type == WT_NONE)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->wv = bind_gguf_matrix(gf, t);
  if (L->wv.type == WT_NONE)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->wo = bind_gguf_matrix(gf, t);
  if (L->wo.type == WT_NONE)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->w_gate = bind_gguf_matrix(gf, t);
  if (L->w_gate.type == WT_NONE)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->w_up = bind_gguf_matrix(gf, t);
  if (L->w_up.type == WT_NONE)
    return -1;

  snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
  t = gguf_find(gf, name);
  if (!t)
    return -1;
  L->w_down = bind_gguf_matrix(gf, t);
  if (L->w_down.type == WT_NONE)
    return -1;

  return 0;
}

int gguf_loader_open(const char *path, Weights *w, const WeightsConfigJson *cfg) {
  GgufFile *gf = calloc(1, sizeof(GgufFile));
  if (!gf)
    return -1;
  if (gguf_open(path, gf) != 0) {
    fprintf(stderr, "gguf: failed to open %s\n", path);
    free(gf);
    return -1;
  }

  memset(w, 0, sizeof(*w));
  w->backing = gf;
  w->qk_interleaved = 1;

  const GgufTensor *t = gguf_find(gf, "token_embd.weight");
  if (!t) {
    fprintf(stderr, "gguf: missing tensor token_embd.weight\n");
    goto gguf_loader_open_fail;
  }
  w->token_emb = bind_gguf_matrix(gf, t);
  if (w->token_emb.type == WT_NONE) {
    fprintf(stderr, "gguf: failed to bind token_embd.weight\n");
    goto gguf_loader_open_fail;
  }

  t = gguf_find(gf, "output_norm.weight");
  if (!t) {
    fprintf(stderr, "gguf: missing tensor output_norm.weight\n");
    goto gguf_loader_open_fail;
  }
  w->rms_final = bind_gguf_norm(gf, t);
  if (!w->rms_final) {
    fprintf(stderr, "gguf: failed to bind output_norm.weight\n");
    goto gguf_loader_open_fail;
  }

  for (int l = 0; l < cfg->num_layers; l++) {
    if (bind_gguf_layer(w, gf, l) != 0) {
      fprintf(stderr, "gguf: failed to bind layer %d\n", l);
      goto gguf_loader_open_fail;
    }
  }

  return 0;

gguf_loader_open_fail:
  free_weights(w, cfg->num_layers);
  return -1;
}
