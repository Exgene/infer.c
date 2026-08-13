#include "json.h"
#include "stdbool.h"

typedef struct WeightsConfigJson {
  int hidden_size, num_layers, num_heads, num_kv_heads, head_dim;
  int intermediate_size, vocab_size, max_seq_len;
  int bos_id, eos_ids[8], num_eos;
  float rms_norm_eps, rope_theta;
  bool tie_word_embeddings;
} WeightsConfigJson;

static int json_get_int(struct json, const char *, int *);

static int json_get_float(struct json, const char *, float *);

int load_config_json(const char *, WeightsConfigJson *);
