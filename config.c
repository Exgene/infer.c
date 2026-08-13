#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int json_get_int(struct json obj, const char *key, int *out) {
  struct json v = json_object_get(obj, key);
  if (!json_exists(v))
    return -1;
  *out = json_int(v);
  return 0;
}

static int json_get_float(struct json obj, const char *key, float *out) {
  struct json v = json_object_get(obj, key);
  if (!json_exists(v))
    return -1;
  *out = (float)json_double(v);
  return 0;
}

int load_config_json(const char *path, WeightsConfigJson *cfg) {
  memset(cfg, 0, sizeof(*cfg));

  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return -1;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    perror("fseek");
    fclose(f);
    return -1;
  }
  long sz = ftell(f);
  if (sz < 0) {
    perror("ftell");
    fclose(f);
    return -1;
  }
  rewind(f);

  char *buf = malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return -1;
  }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    perror("fread config");
    free(buf);
    fclose(f);
    return -1;
  }
  buf[sz] = '\0';
  fclose(f);

  if (!json_valid(buf)) {
    fprintf(stderr, "invalid config json: %s\n", path);
    free(buf);
    return -1;
  }

  struct json root = json_parse(buf);

  if (json_get_int(root, "hidden_size", &cfg->hidden_size) ||
      json_get_int(root, "num_hidden_layers", &cfg->num_layers) ||
      json_get_int(root, "num_attention_heads", &cfg->num_heads) ||
      json_get_int(root, "num_key_value_heads", &cfg->num_kv_heads) ||
      json_get_int(root, "intermediate_size", &cfg->intermediate_size) ||
      json_get_int(root, "vocab_size", &cfg->vocab_size) ||
      json_get_int(root, "max_position_embeddings", &cfg->max_seq_len) ||
      json_get_int(root, "bos_token_id", &cfg->bos_id) ||
      json_get_float(root, "rms_norm_eps", &cfg->rms_norm_eps) ||
      json_get_float(root, "rope_theta", &cfg->rope_theta)) {
    fprintf(stderr, "config missing required field\n");
    free(buf);
    return -1;
  }

  struct json head_dim = json_object_get(root, "head_dim");
  if (json_exists(head_dim))
    cfg->head_dim = json_int(head_dim);
  else if (cfg->num_heads > 0)
    cfg->head_dim = cfg->hidden_size / cfg->num_heads;

  struct json tie = json_object_get(root, "tie_word_embeddings");
  cfg->tie_word_embeddings = json_exists(tie) && json_bool(tie);

  struct json eos = json_object_get(root, "eos_token_id");
  cfg->num_eos = 0;
  if (json_exists(eos)) {
    if (json_type(eos) == JSON_ARRAY) {
      size_t n = json_array_count(eos);
      if (n > 8)
        n = 8;
      for (size_t i = 0; i < n; i++)
        cfg->eos_ids[cfg->num_eos++] = json_int(json_array_get(eos, i));
    } else {
      cfg->eos_ids[cfg->num_eos++] = json_int(eos);
    }
  }

  free(buf);
  return 0;
}
