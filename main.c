#include "json.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *LOCATION = "./models/llama-3.2-1B-instruct/model.safetensors";
const char *CONFIG_LOCATION = "./models/llama-3.2-1B-instruct/config.json";

bool is_metadata(char *v) {
  if (strcmp(v, "__metadata__") == 0)
    return true;
  else
    return false;
}

typedef struct WeightsMetaData {
  char name[256];
  char dtype[16];
  size_t shape[8];
  int ndim;
  uint64_t offset[2]; // [start, end)
} WeightsMetaData;

typedef struct WeightsConfigJson {
  int hidden_size, num_layers, num_heads, num_kv_heads, head_dim;
  int intermediate_size, vocab_size, max_seq_len;
  int bos_id, eos_ids[8], num_eos;
  float rms_norm_eps, rope_theta;
  bool tie_word_embeddings;
} WeightsConfigJson;

typedef struct {
  WeightsMetaData *data;
  size_t len;
  size_t cap;
} WeightsMetaDataVec;

// Vector to cheap out for now.
void vec_init(WeightsMetaDataVec *v) {
  v->data = NULL;
  v->len = v->cap = 0;
}

int vec_push(WeightsMetaDataVec *v, WeightsMetaData item) {
  if (v->len == v->cap) {
    size_t ncap = v->cap ? v->cap * 2 : 8;
    WeightsMetaData *p = realloc(v->data, ncap * sizeof(*p));
    if (!p)
      return -1;
    v->data = p;
    v->cap = ncap;
  }
  v->data[v->len++] = item;
  return 0;
}

void vec_free(WeightsMetaDataVec *v) {
  free(v->data);
  v->data = NULL;
  v->len = v->cap = 0;
}

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

int main(int argc, char *argv[]) {
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

  FILE *input = fopen(LOCATION, "r");
  WeightsMetaDataVec vector;
  vec_init(&vector);

  if (input == NULL) {
    perror("Error while opening the file at location");
    return EXIT_FAILURE;
  }

  uint64_t header_len = 0;

  size_t read_so_far = fread(&header_len, 1, sizeof(uint64_t), input);

  if (read_so_far < sizeof(uint64_t)) {
    fprintf(stderr, "Error: File is too small to contain a u64 integer.\n");
    fclose(input);
    return EXIT_FAILURE;
  }

  if (fseek(input, (long)8, SEEK_SET) != 0) {
    perror("Error while seeking");
    fclose(input);
    return EXIT_FAILURE;
  }

  char *data = malloc(header_len);
  fread(data, 1, header_len, input);

  if (!json_valid(data)) {
    perror("Invalid JSON");
    fclose(input);
    return EXIT_FAILURE;
  }

  struct json parsed = json_parse(data);

  for (struct json j = json_first(parsed); json_exists(j); j = json_next(j)) {
    struct json key = j;
    struct json value = json_next(j);

    char key_str[256];

    json_string_copy(key, key_str, sizeof(key_str));

    if (!is_metadata(key_str)) {
      WeightsMetaData md = {0};

      snprintf(md.name, sizeof(md.name), "%s", key_str);
      struct json dtype = json_object_get(value, "dtype");
      json_string_copy(dtype, md.dtype, sizeof(md.dtype));

      struct json shape = json_object_get(value, "shape");
      md.ndim = (int)json_array_count(shape);
      for (size_t i = 0; i < (size_t)md.ndim; i++)
        md.shape[i] = (size_t)json_uint64(json_array_get(shape, i));

      struct json off = json_object_get(value, "data_offsets");
      md.offset[0] = json_uint64(json_array_get(off, 0));
      md.offset[1] = json_uint64(json_array_get(off, 1));

      printf("name=%s, dtype=%s, ndim=%d \n", md.name, md.dtype, md.ndim);
      printf("-------------\n");

      vec_push(&vector, md);
    }

    j = value;
  }

  fclose(input);
  return EXIT_SUCCESS;
}
