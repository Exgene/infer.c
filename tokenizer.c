#include "tokenizer.h"
#include "json.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_MAX 1024
#define CP_MAP 384

static int cp_to_byte[CP_MAP];

static void build_byte_maps(void) {
  for (int i = 0; i < CP_MAP; i++)
    cp_to_byte[i] = -1;

  int bs[256];
  int cs[256];
  int nbs = 0;
  unsigned char in_bs[256];
  memset(in_bs, 0, sizeof(in_bs));

  for (int b = '!'; b <= '~'; b++) {
    bs[nbs] = b;
    in_bs[b] = 1;
    nbs++;
  }
  for (int b = 0xA1; b <= 0xAC; b++) {
    bs[nbs] = b;
    in_bs[b] = 1;
    nbs++;
  }
  for (int b = 0xAE; b <= 0xFF; b++) {
    bs[nbs] = b;
    in_bs[b] = 1;
    nbs++;
  }
  for (int i = 0; i < nbs; i++)
    cs[i] = bs[i];

  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (in_bs[b])
      continue;
    bs[nbs] = b;
    cs[nbs] = 256 + n;
    nbs++;
    n++;
  }

  for (int i = 0; i < nbs; i++) {
    if (cs[i] >= 0 && cs[i] < CP_MAP)
      cp_to_byte[cs[i]] = bs[i];
  }
}

static int utf8_next(const unsigned char *s, size_t len, size_t *i,
                     uint32_t *cp) {
  if (*i >= len)
    return -1;
  unsigned char c = s[*i];
  if (c < 0x80) {
    *cp = c;
    *i += 1;
    return 0;
  }
  if ((c & 0xE0) == 0xC0 && *i + 1 < len) {
    *cp = ((uint32_t)(c & 0x1F) << 6) | (s[*i + 1] & 0x3F);
    *i += 2;
    return 0;
  }
  if ((c & 0xF0) == 0xE0 && *i + 2 < len) {
    *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[*i + 1] & 0x3F) << 6) |
          (s[*i + 2] & 0x3F);
    *i += 3;
    return 0;
  }
  if ((c & 0xF8) == 0xF0 && *i + 3 < len) {
    *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[*i + 1] & 0x3F) << 12) |
          ((uint32_t)(s[*i + 2] & 0x3F) << 6) | (s[*i + 3] & 0x3F);
    *i += 4;
    return 0;
  }
  *cp = c;
  *i += 1;
  return 0;
}

static int is_special_piece(const char *p) {
  size_t n = strlen(p);
  return n >= 4 && p[0] == '<' && p[1] == '|' && p[n - 2] == '|' &&
         p[n - 1] == '>';
}

static char *dup_str(const char *s) {
  size_t n = strlen(s);
  char *p = malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n + 1);
  return p;
}

static int set_token(Tokenizer *t, int id, const char *piece) {
  if (id < 0 || id >= t->vocab_size)
    return -1;
  free(t->id_to_token[id]);
  t->id_to_token[id] = dup_str(piece);
  return t->id_to_token[id] ? 0 : -1;
}

int tokenizer_load(const char *path, int vocab_size, Tokenizer *t) {
  memset(t, 0, sizeof(*t));
  build_byte_maps();

  if (vocab_size <= 0)
    vocab_size = 128256;

  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return -1;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long sz = ftell(f);
  if (sz < 0) {
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
    free(buf);
    fclose(f);
    return -1;
  }
  buf[sz] = '\0';
  fclose(f);

  if (!json_validn(buf, (size_t)sz)) {
    fprintf(stderr, "invalid tokenizer json\n");
    free(buf);
    return -1;
  }

  t->vocab_size = vocab_size;
  t->id_to_token = calloc((size_t)t->vocab_size, sizeof(char *));
  if (!t->id_to_token) {
    free(buf);
    return -1;
  }

  struct json root = json_parsen(buf, (size_t)sz);
  struct json model = json_object_get(root, "model");
  struct json vocab = json_object_get(model, "vocab");

  for (struct json j = json_first(vocab); json_exists(j); j = json_next(j)) {
    struct json key = j;
    struct json value = json_next(j);
    char piece[TOKEN_MAX];
    json_string_copy(key, piece, sizeof(piece));
    int id = json_int(value);
    if (id >= 0 && id < t->vocab_size) {
      if (set_token(t, id, piece) != 0) {
        free(buf);
        tokenizer_free(t);
        return -1;
      }
    }
    j = value;
  }

  struct json added = json_object_get(root, "added_tokens");
  for (struct json j = json_first(added); json_exists(j); j = json_next(j)) {
    int id = json_int(json_object_get(j, "id"));
    char piece[TOKEN_MAX];
    json_string_copy(json_object_get(j, "content"), piece, sizeof(piece));
    if (id >= 0 && id < t->vocab_size) {
      if (set_token(t, id, piece) != 0) {
        free(buf);
        tokenizer_free(t);
        return -1;
      }
    }
  }

  free(buf);
  return 0;
}

void tokenizer_free(Tokenizer *t) {
  if (!t || !t->id_to_token)
    return;
  for (int i = 0; i < t->vocab_size; i++)
    free(t->id_to_token[i]);
  free(t->id_to_token);
  t->id_to_token = NULL;
  t->vocab_size = 0;
}

const char *tokenizer_lookup(const Tokenizer *t, int id) {
  if (!t || id < 0 || id >= t->vocab_size || !t->id_to_token[id])
    return "";
  return t->id_to_token[id];
}

int tokenizer_decode_id(const Tokenizer *t, int id, char *out, int cap) {
  const char *piece = tokenizer_lookup(t, id);
  if (!piece[0] || is_special_piece(piece) || cap <= 0)
    return 0;

  const unsigned char *s = (const unsigned char *)piece;
  size_t len = strlen(piece);
  size_t i = 0;
  int n = 0;
  while (i < len && n < cap) {
    uint32_t cp = 0;
    if (utf8_next(s, len, &i, &cp) != 0)
      break;
    int b = -1;
    if (cp < (uint32_t)CP_MAP)
      b = cp_to_byte[cp];
    if (b < 0)
      continue;
    out[n++] = (char)b;
  }
  if (n < cap)
    out[n] = '\0';
  else if (cap > 0)
    out[cap - 1] = '\0';
  return n;
}
