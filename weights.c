#include "weights.h"

#include "gguf.h"

#include <stdlib.h>
#include <string.h>

static void free_layer(Layer *layer) {
  free(layer->rms_att);
  free(layer->rms_ffn);
  weight_free(&layer->wq);
  weight_free(&layer->wk);
  weight_free(&layer->wv);
  weight_free(&layer->wo);
  weight_free(&layer->w_gate);
  weight_free(&layer->w_up);
  weight_free(&layer->w_down);
  memset(layer, 0, sizeof(*layer));
}

void free_weights(Weights *w, int num_layers) {
  if (w == NULL)
    return;

  weight_free(&w->token_emb);
  free(w->rms_final);

  for (int l = 0; l < num_layers; l++)
    free_layer(&w->layers[l]);

  if (w->backing) {
    gguf_close((GgufFile *)w->backing);
    free(w->backing);
  }

  memset(w, 0, sizeof(*w));
}
