#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

typedef struct WeightsMetaData {
  char name[256];
  char dtype[16];
  size_t shape[8];
  int ndim;
  uint64_t
      offset[2]; // [start, end) in bytes, from the start of the weight blob
} WeightsMetaData;

typedef struct {
  WeightsMetaData *data;
  size_t len;
  size_t cap;
} WeightsMetaDataVec;

typedef struct {
  void *map;
  size_t map_size;
  uint8_t *blob;
  size_t blob_size;
  WeightsMetaDataVec tensors;
} SafeTensors;

void vec_init(WeightsMetaDataVec *);
int vec_push(WeightsMetaDataVec *, WeightsMetaData);
void vec_free(WeightsMetaDataVec *);

int safetensors_open(const char *, SafeTensors *);
void safetensors_close(SafeTensors *);

const WeightsMetaData *safetensors_find(const SafeTensors *, const char *);
const void *safetensors_ptr(const SafeTensors *, const WeightsMetaData *);

#endif
