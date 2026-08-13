#include "safetensors.h"
#include "json.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

static int is_metadata(const char *v) { return strcmp(v, "__metadata__") == 0; }

static int parse_header_json(const char *json, size_t len,
                             WeightsMetaDataVec *out) {
  if (!json_validn(json, len)) {
    fprintf(stderr, "invalid safetensors header json\n");
    return -1;
  }

  struct json parsed = json_parsen(json, len);

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

      if (vec_push(out, md) != 0)
        return -1;
    }

    j = value;
  }

  return 0;
}

int safetensors_open(const char *path, SafeTensors *out) {
  memset(out, 0, sizeof(*out));
  vec_init(&out->tensors);

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
  if (st.st_size < (off_t)sizeof(uint64_t)) {
    fprintf(stderr, "Error: File is too small to contain a u64 integer.\n");
    close(fd);
    return -1;
  }

  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    perror("mmap");
    return -1;
  }

  uint64_t header_len = 0;
  memcpy(&header_len, map, sizeof(header_len));
  if (sizeof(header_len) + header_len > (size_t)st.st_size) {
    fprintf(stderr, "Error: header length is larger than the file\n");
    munmap(map, (size_t)st.st_size);
    return -1;
  }

  const char *json = (const char *)map + sizeof(header_len);
  if (parse_header_json(json, header_len, &out->tensors) != 0) {
    munmap(map, (size_t)st.st_size);
    vec_free(&out->tensors);
    return -1;
  }

  out->map = map;
  out->map_size = (size_t)st.st_size;
  out->blob = (uint8_t *)map + sizeof(header_len) + header_len;
  out->blob_size = out->map_size - sizeof(header_len) - (size_t)header_len;
  return 0;
}

void safetensors_close(SafeTensors *st) {
  if (!st)
    return;
  if (st->map && st->map_size)
    munmap(st->map, st->map_size);
  vec_free(&st->tensors);
  memset(st, 0, sizeof(*st));
}

const WeightsMetaData *safetensors_find(const SafeTensors *st,
                                        const char *name) {
  if (!st || !name)
    return NULL;
  for (size_t i = 0; i < st->tensors.len; i++) {
    if (strcmp(st->tensors.data[i].name, name) == 0)
      return &st->tensors.data[i];
  }
  return NULL;
}

const void *safetensors_ptr(const SafeTensors *st, const WeightsMetaData *md) {
  if (!st || !st->blob || !md)
    return NULL;
  if (md->offset[0] > md->offset[1] || md->offset[1] > st->blob_size)
    return NULL;
  return st->blob + md->offset[0];
}
