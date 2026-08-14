#include "weights.h"
#include "safetensors.h"
#include <stdint.h>
#include <stdio.h>
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
    if (p == NULL) {
      printf("Not found: %s\n", md->name);
      return -1;
    }

    int layer = 0;
    char rest[127];

    if (strcmp("model.embed_tokens.weight", md->name) == 0) {
      w->token_emb = p;
    } else if (strcmp(md->name, "model.norm.weight") == 0) {
      w->rms_final = p;
    } else if (sscanf(md->name, "model.layers.%d.%126s", &layer, rest) == 2) {
      if (layer < 0 || layer >= cfg->num_layers) {
        return -1;
      }

      if (strcmp(rest, "self_attn.k_proj.weight") == 0) {
        w->layers[layer].wk = p;
      } else if (strcmp(rest, "self_attn.q_proj.weight") == 0) {
        w->layers[layer].wq = p;
      } else if (strcmp(rest, "self_attn.v_proj.weight") == 0) {
        w->layers[layer].wv = p;
      } else if (strcmp(rest, "self_attn.o_proj.weight") == 0) {
        w->layers[layer].wo = p;
      } else if (strcmp(rest, "post_attention_layernorm.weight") == 0) {
        w->layers[layer].rms_ffn = p;
      } else if (strcmp(rest, "input_layernorm.weight") == 0) {
        w->layers[layer].rms_att = p;
      } else if (strcmp(rest, "mlp.down_proj.weight") == 0) {
        w->layers[layer].w_down = p;
      } else if (strcmp(rest, "mlp.up_proj.weight") == 0) {
        w->layers[layer].w_up = p;
      } else if (strcmp(rest, "mlp.gate_proj.weight") == 0) {
        w->layers[layer].w_gate = p;
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
