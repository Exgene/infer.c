#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct WeightsMetaData {
  char name[256];
  char dtype[16];
  size_t shape[8];
  int ndim;
  uint64_t offset[2]; // [start, end)
} WeightsMetaData;

typedef struct {
  WeightsMetaData *data;
  size_t len;
  size_t cap;
} WeightsMetaDataVec;

void vec_init(WeightsMetaDataVec *);

int vec_push(WeightsMetaDataVec *, WeightsMetaData);

void vec_free(WeightsMetaDataVec *);
