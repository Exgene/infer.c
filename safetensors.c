#include "safetensors.h"

void vec_init(WeightsMetaDataVec *v) {
  v->data = NULL;
  v->len = v->cap = 0;
}

int vec_push(WeightsMetaDataVec *v, WeightsMetaData item) {
  if (v->len == v->cap) {
    size_t ncap = v->cap ? v->cap * 2 : 8;
    WeightsMetaData *p = (WeightsMetaData *)realloc(v->data, ncap * sizeof(*p));
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

