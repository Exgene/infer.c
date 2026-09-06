#include "loader.h"

#include "gguf_loader.h"
#include "safetensors_loader.h"

#include <stdio.h>
#include <string.h>

LoaderType loader_detect(const char *path) {
  size_t n = strlen(path);
  if (n >= 12 && strcmp(path + n - 12, ".safetensors") == 0)
    return LOADER_SAFETENSORS;
  if (n >= 5 && strcmp(path + n - 5, ".gguf") == 0)
    return LOADER_GGUF;
  return LOADER_UNKNOWN;
}

int loader_open(const char *path, Weights *w, const WeightsConfigJson *cfg) {
  switch (loader_detect(path)) {
  case LOADER_SAFETENSORS:
    return safetensors_loader_open(path, w, cfg);
  case LOADER_GGUF:
    return gguf_loader_open(path, w, cfg);
  default:
    fprintf(stderr, "loader: unknown format for %s\n", path);
    return -1;
  }
}

void loader_close(Weights *w, int num_layers) {
  free_weights(w, num_layers);
}
