#include "gguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum GgufValueType {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
};

typedef struct {
  const uint8_t *p;
  const uint8_t *end;
} GgufReader;

static int gr_ok(const GgufReader *r, size_t n) {
  return (size_t)(r->end - r->p) >= n;
}

static int gr_u8(GgufReader *r, uint8_t *out) {
  if (!gr_ok(r, 1))
    return -1;
  *out = *r->p++;
  return 0;
}

static int gr_u32(GgufReader *r, uint32_t *out) {
  if (!gr_ok(r, 4))
    return -1;
  memcpy(out, r->p, 4);
  r->p += 4;
  return 0;
}

static int gr_u64(GgufReader *r, uint64_t *out) {
  if (!gr_ok(r, 8))
    return -1;
  memcpy(out, r->p, 8);
  r->p += 8;
  return 0;
}

static int gr_string(GgufReader *r, char *out, size_t out_cap) {
  uint64_t len = 0;
  if (gr_u64(r, &len) != 0)
    return -1;
  if (!gr_ok(r, len))
    return -1;
  if (len + 1 > out_cap)
    return -1;
  memcpy(out, r->p, (size_t)len);
  out[len] = '\0';
  r->p += (size_t)len;
  return 0;
}

static int gr_skip_value(GgufReader *r, uint32_t vtype) {
  switch (vtype) {
  case GGUF_TYPE_UINT8:
  case GGUF_TYPE_INT8:
  case GGUF_TYPE_BOOL:
    return gr_ok(r, 1) ? (r->p++, 0) : -1;
  case GGUF_TYPE_UINT16:
  case GGUF_TYPE_INT16:
    return gr_ok(r, 2) ? (r->p += 2, 0) : -1;
  case GGUF_TYPE_UINT32:
  case GGUF_TYPE_INT32:
  case GGUF_TYPE_FLOAT32:
    return gr_ok(r, 4) ? (r->p += 4, 0) : -1;
  case GGUF_TYPE_UINT64:
  case GGUF_TYPE_INT64:
  case GGUF_TYPE_FLOAT64:
    return gr_ok(r, 8) ? (r->p += 8, 0) : -1;
  case GGUF_TYPE_STRING: {
    uint64_t len = 0;
    if (gr_u64(r, &len) != 0)
      return -1;
    if (!gr_ok(r, len))
      return -1;
    r->p += (size_t)len;
    return 0;
  }
  case GGUF_TYPE_ARRAY: {
    uint32_t arr_type = 0;
    uint64_t count = 0;
    if (gr_u32(r, &arr_type) != 0 || gr_u64(r, &count) != 0)
      return -1;
    for (uint64_t i = 0; i < count; i++) {
      if (gr_skip_value(r, arr_type) != 0)
        return -1;
    }
    return 0;
  }
  default:
    return -1;
  }
}

static size_t ggml_type_size(uint32_t type, const uint64_t *ne, int ndim) {
  uint64_t n = 1;
  for (int i = 0; i < ndim; i++)
    n *= ne[i];

  switch (type) {
  case GGML_TYPE_F32:
    return (size_t)n * 4;
  case GGML_TYPE_F16:
    return (size_t)n * 2;
  case GGML_TYPE_Q4_0:
    return (size_t)((n / 32) * 18);
  case GGML_TYPE_Q4_K:
    return (size_t)((n / 256) * 144);
  case GGML_TYPE_Q6_K:
    return (size_t)((n / 256) * 210);
  case GGML_TYPE_Q8_0:
    return (size_t)((n / 32) * 34);
  default:
    return 0;
  }
}

static size_t align_up(size_t n, size_t align) {
  return (n + align - 1) & ~(align - 1);
}

int gguf_open(const char *path, GgufFile *out) {
  memset(out, 0, sizeof(*out));

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    perror(path);
    return -1;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    perror("fstat");
    close(fd);
    return -1;
  }

  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    perror("mmap");
    return -1;
  }

  GgufReader r = {(const uint8_t *)map,
                  (const uint8_t *)map + (size_t)st.st_size};

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_kv = 0;
  if (gr_u32(&r, &magic) != 0 || magic != GGUF_MAGIC) {
    fprintf(stderr, "gguf: bad magic\n");
    goto fail;
  }
  if (gr_u32(&r, &version) != 0 || version != GGUF_VERSION) {
    fprintf(stderr, "gguf: unsupported version %u\n", version);
    goto fail;
  }
  if (gr_u64(&r, &n_tensors) != 0 || gr_u64(&r, &n_kv) != 0)
    goto fail;

  for (uint64_t i = 0; i < n_kv; i++) {
    char key[256];
    uint32_t vtype = 0;
    if (gr_string(&r, key, sizeof(key)) != 0 || gr_u32(&r, &vtype) != 0)
      goto fail;
    if (gr_skip_value(&r, vtype) != 0)
      goto fail;
  }

  out->tensors = calloc((size_t)n_tensors, sizeof(GgufTensor));
  if (!out->tensors)
    goto fail;

  for (uint64_t i = 0; i < n_tensors; i++) {
    GgufTensor *t = &out->tensors[out->n_tensors++];
    uint32_t ndim = 0;
    if (gr_string(&r, t->name, sizeof(t->name)) != 0 ||
        gr_u32(&r, &ndim) != 0 || ndim == 0 || ndim > 4)
      goto fail;
    t->ndim = (int)ndim;
    for (uint32_t d = 0; d < ndim; d++) {
      if (gr_u64(&r, &t->dims[d]) != 0)
        goto fail;
    }
    if (gr_u32(&r, &t->type) != 0 || gr_u64(&r, &t->offset) != 0)
      goto fail;
    t->nbytes = ggml_type_size(t->type, t->dims, t->ndim);
    if (t->nbytes == 0) {
      fprintf(stderr, "gguf: unsupported type %u for %s\n", t->type, t->name);
      goto fail;
    }
  }

  out->alignment = 32;
  out->data_offset =
      align_up((size_t)(r.p - (const uint8_t *)map), out->alignment);

  size_t max_end = 0;
  for (int i = 0; i < out->n_tensors; i++) {
    GgufTensor *t = &out->tensors[i];
    size_t end = (size_t)t->offset + t->nbytes;
    if (end > max_end)
      max_end = end;
  }
  if (out->data_offset + max_end > (size_t)st.st_size) {
    fprintf(stderr, "gguf: truncated tensor data\n");
    goto fail;
  }

  out->map = map;
  out->map_size = (size_t)st.st_size;
  out->data = (uint8_t *)map + out->data_offset;
  return 0;

fail:
  fprintf(stderr, "gguf: failed to parse %s\n", path);
  free(out->tensors);
  if (map)
    munmap(map, (size_t)st.st_size);
  memset(out, 0, sizeof(*out));
  return -1;
}

void gguf_close(GgufFile *gf) {
  if (!gf)
    return;
  if (gf->map && gf->map_size)
    munmap(gf->map, gf->map_size);
  free(gf->tensors);
  memset(gf, 0, sizeof(*gf));
}

const GgufTensor *gguf_find(const GgufFile *gf, const char *name) {
  if (!gf || !name)
    return NULL;
  for (int i = 0; i < gf->n_tensors; i++) {
    if (strcmp(gf->tensors[i].name, name) == 0)
      return &gf->tensors[i];
  }
  return NULL;
}

const void *gguf_tensor_data(const GgufFile *gf, const GgufTensor *t) {
  if (!gf || !t)
    return NULL;
  return gf->data + t->offset;
}
