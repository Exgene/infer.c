#include "safetensors_loader.h"

#include "ops.h"
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Weight load_bf16_matrix(const uint16_t *src, int rows, int cols) {
  size_t n = (size_t)rows * cols;
  float *dst = malloc(n * sizeof(float));
  if (!dst)
    return (Weight){0};
  for (size_t i = 0; i < n; i++)
    dst[i] = bf16_to_float32(src[i]);
  return weight_make_f32(dst, rows, cols, 1);
}

static float *load_bf16_vector(const uint16_t *src, size_t n) {
  float *dst = malloc(n * sizeof(float));
  if (!dst)
    return NULL;
  for (size_t i = 0; i < n; i++)
    dst[i] = bf16_to_float32(src[i]);
  return dst;
}

static int bind_safetensors_tensor(Weights *w, const WeightsConfigJson *cfg,
                                   const char *name, const uint16_t *p,
                                   size_t n) {
  int layer = 0;
  char rest[127] = {0};
  if (strcmp(name, "model.embed_tokens.weight") == 0) {
    w->token_emb = load_bf16_matrix(p, cfg->vocab_size, cfg->hidden_size);
    return w->token_emb.type == WT_NONE ? -1 : 0;
  }
  if (strcmp(name, "model.norm.weight") == 0) {
    w->rms_final = load_bf16_vector(p, n);
    return w->rms_final ? 0 : -1;
  }
  if (sscanf(name, "model.layers.%d.%126s", &layer, (char *)rest) != 2)
    return -1;
  if (layer < 0 || layer >= cfg->num_layers)
    return -1;

  Layer *L = &w->layers[layer];
  int hidden = cfg->hidden_size;
  int kv_dim = cfg->num_kv_heads * cfg->head_dim;
  int inter = cfg->intermediate_size;

  if (strcmp(rest, "self_attn.k_proj.weight") == 0)
    L->wk = load_bf16_matrix(p, kv_dim, hidden);
  else if (strcmp(rest, "self_attn.q_proj.weight") == 0)
    L->wq = load_bf16_matrix(p, hidden, hidden);
  else if (strcmp(rest, "self_attn.v_proj.weight") == 0)
    L->wv = load_bf16_matrix(p, kv_dim, hidden);
  else if (strcmp(rest, "self_attn.o_proj.weight") == 0)
    L->wo = load_bf16_matrix(p, hidden, hidden);
  else if (strcmp(rest, "post_attention_layernorm.weight") == 0)
    L->rms_ffn = load_bf16_vector(p, n);
  else if (strcmp(rest, "input_layernorm.weight") == 0)
    L->rms_att = load_bf16_vector(p, n);
  else if (strcmp(rest, "mlp.down_proj.weight") == 0)
    L->w_down = load_bf16_matrix(p, hidden, inter);
  else if (strcmp(rest, "mlp.up_proj.weight") == 0)
    L->w_up = load_bf16_matrix(p, inter, hidden);
  else if (strcmp(rest, "mlp.gate_proj.weight") == 0)
    L->w_gate = load_bf16_matrix(p, inter, hidden);
  else
    return -1;

  return 0;
}

int safetensors_loader_open(const char *path, Weights *w,
                            const WeightsConfigJson *cfg) {
  SafeTensors st;
  if (safetensors_open(path, &st) != 0)
    return -1;

  memset(w, 0, sizeof(*w));

  for (size_t i = 0; i < st.tensors.len; i++) {
    WeightsMetaData *md = &st.tensors.data[i];
    const uint16_t *p = (const uint16_t *)safetensors_ptr(&st, md);
    size_t n = 1;
    for (int d = 0; d < md->ndim; d++)
      n *= md->shape[d];

    if (p == NULL) {
      fprintf(stderr, "safetensors: missing tensor %s\n", md->name);
      safetensors_close(&st);
      free_weights(w, cfg->num_layers);
      return -1;
    }

    if (bind_safetensors_tensor(w, cfg, md->name, p, n) != 0) {
      fprintf(stderr, "safetensors: unhandled tensor %s\n", md->name);
      safetensors_close(&st);
      free_weights(w, cfg->num_layers);
      return -1;
    }
  }

  safetensors_close(&st);

  if (w->token_emb.type == WT_NONE || w->rms_final == NULL) {
    free_weights(w, cfg->num_layers);
    return -1;
  }

  return 0;
}
