#ifndef GGUF_H
#define GGUF_H

#include <stddef.h>
#include <stdint.h>

#define GGUF_MAGIC 0x46554747u
#define GGUF_VERSION 3

enum GgmlType {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_0 = 8,
};

typedef struct {
  char name[256];
  int ndim;
  uint64_t dims[4];
  uint32_t type;
  uint64_t offset;
  size_t nbytes;
} GgufTensor;

typedef struct {
  void *map;
  size_t map_size;
  uint8_t *data;
  size_t data_offset;
  size_t alignment;
  GgufTensor *tensors;
  int n_tensors;
} GgufFile;

int gguf_open(const char *path, GgufFile *out);
void gguf_close(GgufFile *gf);
const GgufTensor *gguf_find(const GgufFile *gf, const char *name);
const void *gguf_tensor_data(const GgufFile *gf, const GgufTensor *t);

#endif
