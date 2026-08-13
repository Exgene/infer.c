#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "json.h"
#include "safetensors.h"

const char *LOCATION = "./models/llama-3.2-1B-instruct/model.safetensors";
const char *CONFIG_LOCATION = "./models/llama-3.2-1B-instruct/config.json";

bool is_metadata(char *v) {
  if (strcmp(v, "__metadata__") == 0)
    return true;
  else
    return false;
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
