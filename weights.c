#include "weights.h"
#include "ops.h"
#include "safetensors.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bind_weights(const SafeTensors *st, const WeightsConfigJson *cfg,
                 Weights *w) {
  if (w == NULL || st == NULL || cfg == NULL) {
    return -1;
  }

  memset(w, 0, sizeof(*w));
  for (int i = 0; i < (int)st->tensors.len; i++) {
    WeightsMetaData *md = &st->tensors.data[i];

    const uint16_t *p = (const uint16_t *)safetensors_ptr(st, md);
    size_t n = 1;
    for (int d = 0; d < md->ndim; d++)
      n *= md->shape[d];

    if (p == NULL) {
      printf("Not found: %s\n", md->name);
      return -1;
    }

    int layer = 0;
    char rest[127];

    if (strcmp("model.embed_tokens.weight", md->name) == 0) {
      w->token_emb = convert_tensor(p, n);
    } else if (strcmp(md->name, "model.norm.weight") == 0) {
      w->rms_final = convert_tensor(p, n);
    } else if (sscanf(md->name, "model.layers.%d.%126s", &layer, rest) == 2) {
      if (layer < 0 || layer >= cfg->num_layers) {
        return -1;
      }

      if (strcmp(rest, "self_attn.k_proj.weight") == 0) {
        w->layers[layer].wk = convert_tensor(p, n);
      } else if (strcmp(rest, "self_attn.q_proj.weight") == 0) {
        w->layers[layer].wq = convert_tensor(p, n);
      } else if (strcmp(rest, "self_attn.v_proj.weight") == 0) {
        w->layers[layer].wv = convert_tensor(p, n);
      } else if (strcmp(rest, "self_attn.o_proj.weight") == 0) {
        w->layers[layer].wo = convert_tensor(p, n);
      } else if (strcmp(rest, "post_attention_layernorm.weight") == 0) {
        w->layers[layer].rms_ffn = convert_tensor(p, n);
      } else if (strcmp(rest, "input_layernorm.weight") == 0) {
        w->layers[layer].rms_att = convert_tensor(p, n);
      } else if (strcmp(rest, "mlp.down_proj.weight") == 0) {
        w->layers[layer].w_down = convert_tensor(p, n);
      } else if (strcmp(rest, "mlp.up_proj.weight") == 0) {
        w->layers[layer].w_up = convert_tensor(p, n);
      } else if (strcmp(rest, "mlp.gate_proj.weight") == 0) {
        w->layers[layer].w_gate = convert_tensor(p, n);
      } else {
        return -1;
      }
    } else
      return -1;
  }

  if (w->token_emb == NULL || w->rms_final == NULL) {
    return -1;
  }

  return 0;
}

float *convert_tensor(const uint16_t *src, size_t n) {
  float *dst = malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) {
    dst[i] = bf16_to_float32(src[i]);
  }
  return dst;
}

static void free_layer(Layer *layer) {
  free(layer->rms_att);
  free(layer->wq);
  free(layer->wk);
  free(layer->wv);
  free(layer->wo);
  free(layer->rms_ffn);
  free(layer->w_gate);
  free(layer->w_up);
  free(layer->w_down);
  memset(layer, 0, sizeof(*layer));
}

void free_weights(Weights *w, int num_layers) {
  if (w == NULL)
    return;

  free(w->token_emb);
  free(w->rms_final);

  for (int l = 0; l < num_layers; l++)
    free_layer(&w->layers[l]);

  memset(w, 0, sizeof(*w));
}
