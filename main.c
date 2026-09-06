#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "loader.h"
#include "ops.h"
#include "prof.h"
#include "tokenizer.h"
#include "weights.h"
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

const char *DEFAULT_MODEL =
    "./models/llama-3.2-1B-instruct/model.safetensors";
const char *CONFIG_LOCATION = "./models/llama-3.2-1B-instruct/config.json";
const char *TOKENIZER_LOCATION =
    "./models/llama-3.2-1B-instruct/tokenizer.json";

static const char *arg_get_str(int argc, char **argv, const char *flag,
                               const char *fallback) {
  for (int i = 1; i + 1 < argc; i++) {
    if (strcmp(argv[i], flag) == 0)
      return argv[i + 1];
  }
  return fallback;
}

static int arg_has_flag(int argc, char **argv, const char *flag) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], flag) == 0)
      return 1;
  }
  return 0;
}

static int arg_get_int(int argc, char **argv, const char *flag, int fallback) {
  for (int i = 1; i + 1 < argc; i++) {
    if (strcmp(argv[i], flag) == 0) {
      int v = atoi(argv[i + 1]);
      return v > 0 ? v : fallback;
    }
  }
  return fallback;
}

static int prof_thread_count(void) {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int main(int argc, char *argv[]) {
  int benchmark = arg_has_flag(argc, argv, "--benchmark");
  int profile = benchmark || arg_has_flag(argc, argv, "--profile");
  int gen_tokens = arg_get_int(argc, argv, "-n", 128);
  const char *model_path = arg_get_str(argc, argv, "--model", DEFAULT_MODEL);

  if (!benchmark)
    srand((unsigned)time(NULL));

  ProfTime t_load = prof_now();

  WeightsConfigJson config;
  if (load_config_json(CONFIG_LOCATION, &config) != 0)
    return EXIT_FAILURE;

  if (!benchmark) {
    printf("config: hidden=%d layers=%d heads=%d kv=%d head_dim=%d inter=%d "
           "vocab=%d max_seq=%d bos=%d eos0=%d rms=%g theta=%g tie=%d\n",
           config.hidden_size, config.num_layers, config.num_heads,
           config.num_kv_heads, config.head_dim, config.intermediate_size,
           config.vocab_size, config.max_seq_len, config.bos_id,
           config.num_eos ? config.eos_ids[0] : -1, config.rms_norm_eps,
           config.rope_theta, (int)config.tie_word_embeddings);
  }

  Tokenizer tok;
  if (tokenizer_load(TOKENIZER_LOCATION, config.vocab_size, &tok) != 0)
    return EXIT_FAILURE;

  if (!benchmark) {
    fprintf(stderr, "tokenizer: %d ids, bos=%s\n", tok.vocab_size,
            tokenizer_lookup(&tok, config.bos_id));
  }

  // for (size_t i = 0; i < st.tensors.len; i++) {
  //   WeightsMetaData *md = &st.tensors.data[i];
  //   printf("name=%s, dtype=%s, ndim=%d\n", md->name, md->dtype, md->ndim);
  //   printf("-------------\n");
  // }

  // if (emb) {
  //   const void *p = safetensors_ptr(&st, emb);
  //   printf("mmap: file=%zu bytes, blob=%zu bytes\n", st.map_size,
  //   st.blob_size); printf("embed_tokens ptr=%p bytes=%llu dtype=%s\n", p,
  //          (unsigned long long)(emb->offset[1] - emb->offset[0]),
  //          emb->dtype);
  // } else {
  //   fprintf(stderr, "model.embed_tokens.weight not found\n");
  // }

  Weights w;
  if (loader_open(model_path, &w, &config) != 0) {
    fprintf(stderr, "failed to load model: %s\n", model_path);
    return EXIT_FAILURE;
  }

  // we don't really need the metadat about the ST anymore

  double load_sec = prof_elapsed(t_load, prof_now());

  const int max_seq = 128;

  float *logits = malloc(config.vocab_size * sizeof(float));

  float *x = malloc(config.hidden_size * sizeof(float) * max_seq);
  float *xn = malloc(config.hidden_size * sizeof(float) * max_seq);

  float *q = malloc(config.hidden_size * sizeof(float) * max_seq);
  float *k =
      malloc(config.num_kv_heads * config.head_dim * sizeof(float) * max_seq);
  float *v =
      malloc(config.num_kv_heads * config.head_dim * sizeof(float) * max_seq);
  float *attn = malloc(config.hidden_size * sizeof(float) * max_seq);

  float *hb = malloc(config.intermediate_size * sizeof(float) * max_seq);
  float *hb2 = malloc(config.intermediate_size * sizeof(float) * max_seq);
  float *score = malloc(sizeof(float) * max_seq);

  int kv_dim = config.num_kv_heads * config.head_dim;
  size_t kv_cache_bytes =
      (size_t)config.num_layers * max_seq * kv_dim * sizeof(float);
  float *k_cache = malloc(kv_cache_bytes);
  float *v_cache = malloc(kv_cache_bytes);

  int *tokens = malloc(sizeof(int) * max_seq);
  Prob *ps = malloc((size_t)config.vocab_size * sizeof(Prob));

  if (!logits || !x || !xn || !q || !k || !v || !attn || !hb || !hb2 ||
      !score || !k_cache || !v_cache || !tokens || !ps) {
    return EXIT_FAILURE;
  }

  const char *user = benchmark ? "Write a short story about a robot."
                               : "Do you like ice cream?";
  char prompt[4096];

  // I can parse the tokenizer-config.json to get the template. but idw waste
  // time parsing JINJA SHI
  int m =
      snprintf(prompt, sizeof(prompt),
               "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
               "%s"
               "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
               user);

  if (m < 0 || m >= (int)sizeof(prompt))
    return EXIT_FAILURE;

  if (!benchmark)
    fprintf(stderr, "System Prompt + User prompt: %s", prompt);

  int n = tokenizer_encode(&tok, prompt, tokens, max_seq);
  if (n < 0)
    return EXIT_FAILURE;

  if (!benchmark) {
    fprintf(stderr, "encode:");
    for (int i = 0; i < n; i++)
      fprintf(stderr, " %d", tokens[i]);
    fprintf(stderr, "\n");
  }

  float scale = config.rope_scale;
  float low = config.rope_low;
  float high = config.rope_high;
  int origin = config.rope_origin_ctx;
  float temperature = 0.9f;
  int next = 0;

  ProfTime t_prefill = prof_now();

  next = forward_prefill(&config, &w, x, xn, q, k, v, attn, hb, hb2, logits,
                         tokens, n, k_cache, v_cache, max_seq, scale, low, high,
                         origin, score);
  // for (int pos = 0; pos < n; pos++) {
  //   next = forward(&config, &w, x, xn, q, k, v, attn, hb, hb2, logits,
  //                  tokens[pos], pos, k_cache, v_cache, max_seq, scale, low,
  //                  high, origin, score);
  // }
  double prefill_sec = prof_elapsed(t_prefill, prof_now());

  if (!benchmark)
    next = sample_top_p(logits, config.vocab_size, 0.9f, temperature, ps);

  int generated = 0;
  ProfTime t_decode = prof_now();

  if (benchmark) {
    for (int i = 0; i < gen_tokens && n + 1 < max_seq; i++) {
      tokens[n] = next;
      next = forward(&config, &w, x, xn, q, k, v, attn, hb, hb2, logits,
                     tokens[n], n, k_cache, v_cache, max_seq, scale, low, high,
                     origin, score);
      n++;
      generated++;
    }
  } else {
    // existing stuff from main
    for (;;) {
      char piece[1024];
      int nbytes = tokenizer_decode_id(&tok, next, piece, sizeof(piece));
      if (nbytes > 0)
        fwrite(piece, 1, (size_t)nbytes, stdout);
      fflush(stdout);

      int eos = 0;
      for (int i = 0; i < config.num_eos; i++) {
        if (next == config.eos_ids[i])
          eos = 1;
      }
      if (eos || n >= max_seq)
        break;

      tokens[n] = next;
      next = forward(&config, &w, x, xn, q, k, v, attn, hb, hb2, logits,
                     tokens[n], n, k_cache, v_cache, max_seq, scale, low, high,
                     origin, score);
      next = sample_top_p(logits, config.vocab_size, 0.9f, temperature, ps);
      n++;
      generated++;
    }
    printf("\n");
  }

  double decode_sec = prof_elapsed(t_decode, prof_now());

  if (profile)
    prof_report(load_sec, n - generated, prefill_sec, generated, decode_sec,
                prof_thread_count());

  tokenizer_free(&tok);
  loader_close(&w, config.num_layers);
  free(logits);
  free(x);
  free(xn);
  free(q);
  free(k);
  free(v);
  free(attn);
  free(hb);
  free(hb2);
  free(score);
  free(k_cache);
  free(v_cache);
  free(tokens);
  free(ps);
  return EXIT_SUCCESS;
}
