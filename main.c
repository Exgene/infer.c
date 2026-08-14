#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "safetensors.h"
#include "weights.h"

const char *LOCATION = "./models/llama-3.2-1B-instruct/model.safetensors";
const char *CONFIG_LOCATION = "./models/llama-3.2-1B-instruct/config.json";

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  WeightsConfigJson config;
  if (load_config_json(CONFIG_LOCATION, &config) != 0)
    return EXIT_FAILURE;

  printf("config: hidden=%d layers=%d heads=%d kv=%d head_dim=%d inter=%d "
         "vocab=%d max_seq=%d bos=%d eos0=%d rms=%g theta=%g tie=%d\n",
         config.hidden_size, config.num_layers, config.num_heads,
         config.num_kv_heads, config.head_dim, config.intermediate_size,
         config.vocab_size, config.max_seq_len, config.bos_id,
         config.num_eos ? config.eos_ids[0] : -1, config.rms_norm_eps,
         config.rope_theta, (int)config.tie_word_embeddings);

  SafeTensors st;
  if (safetensors_open(LOCATION, &st) != 0)
    return EXIT_FAILURE;

  for (size_t i = 0; i < st.tensors.len; i++) {
    WeightsMetaData *md = &st.tensors.data[i];
    printf("name=%s, dtype=%s, ndim=%d\n", md->name, md->dtype, md->ndim);
    printf("-------------\n");
  }

  const WeightsMetaData *emb =
      safetensors_find(&st, "model.embed_tokens.weight");
  if (emb) {
    const void *p = safetensors_ptr(&st, emb);
    printf("mmap: file=%zu bytes, blob=%zu bytes\n", st.map_size, st.blob_size);
    printf("embed_tokens ptr=%p bytes=%llu dtype=%s\n", p,
           (unsigned long long)(emb->offset[1] - emb->offset[0]), emb->dtype);
  } else {
    fprintf(stderr, "model.embed_tokens.weight not found\n");
  }

  Weights w;
  int error = bind_weights(&st, &config, &w);
  if (error) {
    return EXIT_FAILURE;
  }

  safetensors_close(&st);
  return EXIT_SUCCESS;
}
