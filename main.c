#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "ops.h"
#include "safetensors.h"
#include "tokenizer.h"
#include "weights.h"

const char *LOCATION = "./models/llama-3.2-1B-instruct/model.safetensors";
const char *CONFIG_LOCATION = "./models/llama-3.2-1B-instruct/config.json";
const char *TOKENIZER_LOCATION =
    "./models/llama-3.2-1B-instruct/tokenizer.json";

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

  Tokenizer tok;
  if (tokenizer_load(TOKENIZER_LOCATION, config.vocab_size, &tok) != 0)
    return EXIT_FAILURE;
  fprintf(stderr, "tokenizer: %d ids, bos=%s\n", tok.vocab_size,
          tokenizer_lookup(&tok, config.bos_id));

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

  float *logits = malloc(config.vocab_size * sizeof(float));

  float *x = malloc(config.hidden_size * sizeof(float));
  float *xn = malloc(config.hidden_size * sizeof(float));

  float *q = malloc(config.hidden_size * sizeof(float));
  float *k = malloc(config.num_kv_heads * config.head_dim * sizeof(float));
  float *v = malloc(config.num_kv_heads * config.head_dim * sizeof(float));
  float *attn = malloc(config.hidden_size * sizeof(float));

  float *hb = malloc(config.intermediate_size * sizeof(float));
  float *hb2 = malloc(config.intermediate_size * sizeof(float));

  const int max_seq = 32;
  int kv_dim = config.num_kv_heads * config.head_dim;
  float *k_cache =
      malloc((size_t)config.num_layers * max_seq * kv_dim * sizeof(float));
  float *v_cache =
      malloc((size_t)config.num_layers * max_seq * kv_dim * sizeof(float));

  int tokens[32] = {0};
  int n = tokenizer_encode_bos(&tok, "hello", tokens, max_seq);
  if (n < 0)
    return EXIT_FAILURE;
  fprintf(stderr, "encode:");
  for (int i = 0; i < n; i++)
    fprintf(stderr, " %d", tokens[i]);
  fprintf(stderr, "\n");
  int next = 0;

  fprintf(stderr, "running first forward (this can take a while)...\n");
  for (int pos = 0; pos < n; pos++) {
    next = forward(&config, &w, x, xn, q, k, v, attn, hb, hb2, logits,
                   tokens[pos], pos, k_cache, v_cache, max_seq);
  }
  for (;;) {
    char piece[1024];
    int nbytes = tokenizer_decode_id(&tok, next, piece, sizeof(piece));
    if (nbytes > 0)
      fwrite(piece, 1, (size_t)nbytes, stdout);
    fflush(stdout);
    int eos = 0;
    for (int i = 0; i < config.num_eos; i++)
      if (next == config.eos_ids[i])
        eos = 1;
    if (eos || n >= max_seq)
      break;
    tokens[n] = next;
    next = forward(&config, &w, x, xn, q, k, v, attn, hb, hb2, logits,
                   tokens[n], n, k_cache, v_cache, max_seq);
    n++;
  }
  printf("\n");

  tokenizer_free(&tok);
  safetensors_close(&st);
  return EXIT_SUCCESS;
}
