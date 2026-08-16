#include "tokenizer.h"
#include "json.h"

#include <limits.h>
#include <locale.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

#define TOKEN_MAX 8192
#define CP_MAP 384

static int cp_to_byte[CP_MAP];
static int byte_to_cp[256];

static char *dup_str(const char *s) {
  size_t n = strlen(s);
  char *p = malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n + 1);
  return p;
}

static char *concat_str(const char *a, const char *b) {
  size_t na = strlen(a);
  size_t nb = strlen(b);
  char *p = malloc(na + nb + 1);
  if (!p)
    return NULL;
  memcpy(p, a, na);
  memcpy(p + na, b, nb + 1);
  return p;
}

static unsigned strhash(const char *s) {
  unsigned h = 5381;
  while (*s)
    h = ((h << 5) + h) ^ (unsigned char)*s++;
  return h;
}

static int hmap_init(TokMap *m, int cap) {
  m->cap = cap;
  m->len = 0;
  m->slots = calloc((size_t)cap, sizeof(TokSlot));
  return m->slots ? 0 : -1;
}

static void hmap_free(TokMap *m) {
  if (!m || !m->slots)
    return;
  for (int i = 0; i < m->cap; i++)
    free(m->slots[i].key);
  free(m->slots);
  m->slots = NULL;
  m->cap = 0;
  m->len = 0;
}

static int hmap_put_owned(TokMap *m, char *key, int val);

static int hmap_grow(TokMap *m) {
  TokMap nxt;
  if (hmap_init(&nxt, m->cap * 2) != 0)
    return -1;
  for (int i = 0; i < m->cap; i++) {
    if (!m->slots[i].key)
      continue;
    if (hmap_put_owned(&nxt, m->slots[i].key, m->slots[i].val) != 0) {
      nxt.slots[i].key = NULL;
      hmap_free(&nxt);
      return -1;
    }
    m->slots[i].key = NULL;
  }
  free(m->slots);
  *m = nxt;
  return 0;
}

static int hmap_put_owned(TokMap *m, char *key, int val) {
  if ((m->len + 1) * 2 >= m->cap) {
    if (hmap_grow(m) != 0) {
      free(key);
      return -1;
    }
  }
  unsigned h = strhash(key);
  for (int i = 0; i < m->cap; i++) {
    int j = (int)((h + (unsigned)i) & (unsigned)(m->cap - 1));
    if (!m->slots[j].key) {
      m->slots[j].key = key;
      m->slots[j].val = val;
      m->len++;
      return 0;
    }
    if (strcmp(m->slots[j].key, key) == 0) {
      m->slots[j].val = val;
      free(key);
      return 0;
    }
  }
  free(key);
  return -1;
}

static int hmap_put(TokMap *m, const char *key, int val) {
  char *copy = dup_str(key);
  if (!copy)
    return -1;
  return hmap_put_owned(m, copy, val);
}

static int hmap_get(const TokMap *m, const char *key, int *val) {
  if (!m || !m->slots || !key)
    return 0;
  unsigned h = strhash(key);
  for (int i = 0; i < m->cap; i++) {
    int j = (int)((h + (unsigned)i) & (unsigned)(m->cap - 1));
    if (!m->slots[j].key)
      return 0;
    if (strcmp(m->slots[j].key, key) == 0) {
      *val = m->slots[j].val;
      return 1;
    }
  }
  return 0;
}

static void build_byte_maps(void) {
  for (int i = 0; i < CP_MAP; i++)
    cp_to_byte[i] = -1;
  for (int i = 0; i < 256; i++)
    byte_to_cp[i] = -1;

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
    byte_to_cp[bs[i]] = cs[i];
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

static int utf8_put(char *out, int cap, uint32_t cp) {
  if (cp < 0x80) {
    if (cap < 1)
      return -1;
    out[0] = (char)cp;
    return 1;
  }
  if (cp < 0x800) {
    if (cap < 2)
      return -1;
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    if (cap < 3)
      return -1;
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  if (cap < 4)
    return -1;
  out[0] = (char)(0xF0 | (cp >> 18));
  out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
  out[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

static int is_special_piece(const char *p) {
  size_t n = strlen(p);
  return n >= 4 && p[0] == '<' && p[1] == '|' && p[n - 2] == '|' &&
         p[n - 1] == '>';
}

static int is_ws(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0b ||
         cp == 0x0c || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

static int is_letter(uint32_t cp) {
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'))
    return 1;
  if (cp < 128)
    return 0;
  return iswalpha((wint_t)cp) != 0;
}

static int is_number(uint32_t cp) {
  if (cp >= '0' && cp <= '9')
    return 1;
  if (cp < 128)
    return 0;
  return iswdigit((wint_t)cp) != 0;
}

static int set_token(Tokenizer *t, int id, const char *piece) {
  if (id < 0 || id >= t->vocab_size)
    return -1;
  free(t->id_to_token[id]);
  t->id_to_token[id] = dup_str(piece);
  return t->id_to_token[id] ? 0 : -1;
}

static int special_cmp(const void *a, const void *b) {
  const TokSpecial *sa = a;
  const TokSpecial *sb = b;
  size_t la = strlen(sa->s);
  size_t lb = strlen(sb->s);
  if (la < lb)
    return 1;
  if (la > lb)
    return -1;
  return 0;
}

static int starts_ci(const unsigned char *s, size_t len, size_t pos,
                     const char *lit, size_t *end) {
  size_t n = strlen(lit);
  if (pos + n > len)
    return 0;
  for (size_t i = 0; i < n; i++) {
    unsigned char a = s[pos + i];
    unsigned char b = (unsigned char)lit[i];
    if (a >= 'A' && a <= 'Z')
      a = (unsigned char)(a + 32);
    if (b >= 'A' && b <= 'Z')
      b = (unsigned char)(b + 32);
    if (a != b)
      return 0;
  }
  *end = pos + n;
  return 1;
}

static int try_contraction(const unsigned char *s, size_t len, size_t pos,
                           size_t *end) {
  static const char *lits[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
                               NULL};
  for (int i = 0; lits[i]; i++) {
    if (starts_ci(s, len, pos, lits[i], end))
      return 1;
  }
  return 0;
}

static int try_word(const unsigned char *s, size_t len, size_t pos,
                    size_t *end) {
  if (pos >= len)
    return 0;
  size_t i = pos;
  size_t j = i;
  uint32_t cp = 0;
  if (utf8_next(s, len, &j, &cp) != 0)
    return 0;
  size_t after = pos;
  if (!is_letter(cp) && !is_number(cp) && cp != '\r' && cp != '\n')
    after = j;
  i = after;
  int nlet = 0;
  while (i < len) {
    j = i;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    if (!is_letter(cp))
      break;
    i = j;
    nlet++;
  }
  if (nlet == 0)
    return 0;
  *end = i;
  return 1;
}

static int try_number(const unsigned char *s, size_t len, size_t pos,
                      size_t *end) {
  size_t i = pos;
  int n = 0;
  while (n < 3 && i < len) {
    size_t j = i;
    uint32_t cp = 0;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    if (!is_number(cp))
      break;
    i = j;
    n++;
  }
  if (n == 0)
    return 0;
  *end = i;
  return 1;
}

static int try_punct(const unsigned char *s, size_t len, size_t pos,
                     size_t *end) {
  size_t i = pos;
  if (i < len && s[i] == ' ')
    i++;
  int n = 0;
  while (i < len) {
    size_t j = i;
    uint32_t cp = 0;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    if (is_ws(cp) || is_letter(cp) || is_number(cp))
      break;
    i = j;
    n++;
  }
  if (n == 0)
    return 0;
  while (i < len) {
    size_t j = i;
    uint32_t cp = 0;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    if (cp != '\r' && cp != '\n')
      break;
    i = j;
  }
  *end = i;
  return 1;
}

static int try_newline(const unsigned char *s, size_t len, size_t pos,
                       size_t *end) {
  size_t i = pos;
  size_t last_nl = (size_t)-1;
  while (i < len) {
    size_t j = i;
    uint32_t cp = 0;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    if (!is_ws(cp))
      break;
    if (cp == '\n' || cp == '\r')
      last_nl = j;
    i = j;
  }
  if (last_nl == (size_t)-1)
    return 0;
  *end = last_nl;
  return 1;
}

static int try_ws(const unsigned char *s, size_t len, size_t pos, size_t *end) {
  size_t i = pos;
  int n = 0;
  while (i < len) {
    size_t j = i;
    uint32_t cp = 0;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    if (!is_ws(cp))
      break;
    i = j;
    n++;
  }
  if (n == 0)
    return 0;
  *end = i;
  return 1;
}

static int try_trail_ws(const unsigned char *s, size_t len, size_t pos,
                        size_t *end) {
  size_t e = 0;
  if (!try_ws(s, len, pos, &e))
    return 0;
  if (e != len)
    return 0;
  *end = e;
  return 1;
}

static int split_next(const unsigned char *s, size_t len, size_t pos,
                      size_t *end) {
  if (try_contraction(s, len, pos, end))
    return 1;
  if (try_word(s, len, pos, end))
    return 1;
  if (try_number(s, len, pos, end))
    return 1;
  if (try_punct(s, len, pos, end))
    return 1;
  if (try_newline(s, len, pos, end))
    return 1;
  if (try_trail_ws(s, len, pos, end))
    return 1;
  if (try_ws(s, len, pos, end))
    return 1;
  return 0;
}

static int merge_rank_of(const Tokenizer *t, const char *a, const char *b) {
  size_t na = strlen(a);
  size_t nb = strlen(b);
  char stack[TOKEN_MAX];
  char *key = stack;
  int heap = 0;
  if (na + nb + 2 > sizeof(stack)) {
    key = malloc(na + nb + 2);
    if (!key)
      return INT_MAX;
    heap = 1;
  }
  memcpy(key, a, na);
  key[na] = ' ';
  memcpy(key + na + 1, b, nb + 1);
  int rank = INT_MAX;
  int found = 0;
  if (hmap_get(&t->merge_rank, key, &found))
    rank = found;
  if (heap)
    free(key);
  return rank;
}

static int bpe_encode(const Tokenizer *t, const char *alpha, int *out,
                      int cap) {
  int id = 0;
  if (alpha[0] && hmap_get(&t->piece_to_id, alpha, &id)) {
    if (cap < 1)
      return -1;
    out[0] = id;
    return 1;
  }

  int n = 0;
  int pcap = 0;
  char **ps = NULL;
  const unsigned char *s = (const unsigned char *)alpha;
  size_t len = strlen(alpha);
  size_t i = 0;
  while (i < len) {
    uint32_t cp = 0;
    size_t j = i;
    if (utf8_next(s, len, &j, &cp) != 0)
      break;
    char tmp[8];
    int k = utf8_put(tmp, (int)sizeof(tmp) - 1, cp);
    if (k < 0)
      break;
    tmp[k] = '\0';
    if (n >= pcap) {
      int ncap = pcap ? pcap * 2 : 16;
      char **np = realloc(ps, (size_t)ncap * sizeof(char *));
      if (!np) {
        for (int x = 0; x < n; x++)
          free(ps[x]);
        free(ps);
        return -1;
      }
      ps = np;
      pcap = ncap;
    }
    ps[n] = dup_str(tmp);
    if (!ps[n]) {
      for (int x = 0; x < n; x++)
        free(ps[x]);
      free(ps);
      return -1;
    }
    n++;
    i = j;
  }

  for (;;) {
    int best_rank = INT_MAX;
    int best_i = -1;
    for (int k = 0; k < n - 1; k++) {
      int rank = merge_rank_of(t, ps[k], ps[k + 1]);
      if (rank < best_rank) {
        best_rank = rank;
        best_i = k;
      }
    }
    if (best_i < 0)
      break;
    char *merged = concat_str(ps[best_i], ps[best_i + 1]);
    if (!merged) {
      for (int x = 0; x < n; x++)
        free(ps[x]);
      free(ps);
      return -1;
    }
    free(ps[best_i]);
    free(ps[best_i + 1]);
    ps[best_i] = merged;
    memmove(&ps[best_i + 1], &ps[best_i + 2],
            (size_t)(n - best_i - 2) * sizeof(char *));
    n--;
  }

  if (n > cap) {
    for (int x = 0; x < n; x++)
      free(ps[x]);
    free(ps);
    return -1;
  }
  for (int k = 0; k < n; k++) {
    if (!hmap_get(&t->piece_to_id, ps[k], &out[k])) {
      fprintf(stderr, "missing vocab piece '%s'\n", ps[k]);
      for (int x = 0; x < n; x++)
        free(ps[x]);
      free(ps);
      return -1;
    }
    free(ps[k]);
  }
  free(ps);
  return n;
}

static int bytes_to_alphabet(const unsigned char *s, size_t len, char *out,
                             int cap) {
  int n = 0;
  for (size_t i = 0; i < len; i++) {
    int cp = byte_to_cp[s[i]];
    if (cp < 0)
      return -1;
    int k = utf8_put(out + n, cap - n, (uint32_t)cp);
    if (k < 0)
      return -1;
    n += k;
  }
  if (n >= cap)
    return -1;
  out[n] = '\0';
  return n;
}

static int encode_ordinary(const Tokenizer *t, const unsigned char *s,
                           size_t len, int *out, int cap) {
  size_t pos = 0;
  int nout = 0;
  while (pos < len) {
    size_t end = pos;
    if (!split_next(s, len, pos, &end) || end <= pos) {
      uint32_t cp = 0;
      end = pos;
      if (utf8_next(s, len, &end, &cp) != 0)
        return -1;
      if (end <= pos)
        return -1;
    }
    size_t clen = end - pos;
    char *alpha = malloc(clen * 2 + 1);
    if (!alpha)
      return -1;
    if (bytes_to_alphabet(s + pos, clen, alpha, (int)(clen * 2 + 1)) < 0) {
      free(alpha);
      return -1;
    }
    int k = bpe_encode(t, alpha, out + nout, cap - nout);
    free(alpha);
    if (k < 0)
      return -1;
    nout += k;
    pos = end;
  }
  return nout;
}

static int find_special(const Tokenizer *t, const char *s, size_t len,
                        size_t pos) {
  for (int i = 0; i < t->n_specials; i++) {
    size_t L = strlen(t->specials[i].s);
    if (pos + L <= len && memcmp(s + pos, t->specials[i].s, L) == 0)
      return i;
  }
  return -1;
}

static size_t next_special_pos(const Tokenizer *t, const char *s, size_t pos,
                               size_t len) {
  size_t best = len;
  for (int i = 0; i < t->n_specials; i++) {
    const char *p = strstr(s + pos, t->specials[i].s);
    if (!p)
      continue;
    size_t off = (size_t)(p - s);
    if (off < best)
      best = off;
  }
  return best;
}

int tokenizer_load(const char *path, int vocab_size, Tokenizer *t) {
  memset(t, 0, sizeof(*t));
  t->bos_id = 128000;
  build_byte_maps();
  setlocale(LC_CTYPE, "C.UTF-8");

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

  if (hmap_init(&t->piece_to_id, 1 << 18) != 0) {
    free(buf);
    tokenizer_free(t);
    return -1;
  }
  for (int i = 0; i < t->vocab_size; i++) {
    if (!t->id_to_token[i])
      continue;
    if (hmap_put(&t->piece_to_id, t->id_to_token[i], i) != 0) {
      free(buf);
      tokenizer_free(t);
      return -1;
    }
  }

  int ns = 0;
  for (int i = 0; i < t->vocab_size; i++) {
    if (t->id_to_token[i] && is_special_piece(t->id_to_token[i]))
      ns++;
  }
  t->specials = calloc((size_t)ns, sizeof(TokSpecial));
  if (!t->specials) {
    free(buf);
    tokenizer_free(t);
    return -1;
  }
  t->n_specials = ns;
  int k = 0;
  for (int i = 0; i < t->vocab_size; i++) {
    if (!t->id_to_token[i] || !is_special_piece(t->id_to_token[i]))
      continue;
    t->specials[k].s = t->id_to_token[i];
    t->specials[k].id = i;
    if (strcmp(t->id_to_token[i], "<|begin_of_text|>") == 0)
      t->bos_id = i;
    k++;
  }
  qsort(t->specials, (size_t)t->n_specials, sizeof(TokSpecial), special_cmp);

  if (hmap_init(&t->merge_rank, 1 << 19) != 0) {
    free(buf);
    tokenizer_free(t);
    return -1;
  }
  struct json merges = json_object_get(model, "merges");
  int rank = 0;
  for (struct json j = json_first(merges); json_exists(j); j = json_next(j)) {
    char pair[TOKEN_MAX];
    json_string_copy(j, pair, sizeof(pair));
    if (hmap_put(&t->merge_rank, pair, rank) != 0) {
      free(buf);
      tokenizer_free(t);
      return -1;
    }
    rank++;
  }

  free(buf);
  return 0;
}

void tokenizer_free(Tokenizer *t) {
  if (!t)
    return;
  if (t->id_to_token) {
    for (int i = 0; i < t->vocab_size; i++)
      free(t->id_to_token[i]);
    free(t->id_to_token);
  }
  hmap_free(&t->piece_to_id);
  hmap_free(&t->merge_rank);
  free(t->specials);
  memset(t, 0, sizeof(*t));
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

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int cap) {
  if (!t || !text || !out || cap <= 0)
    return -1;
  const char *s = text;
  size_t len = strlen(text);
  size_t pos = 0;
  int nout = 0;
  while (pos < len) {
    int sp = find_special(t, s, len, pos);
    if (sp >= 0) {
      if (nout >= cap)
        return -1;
      out[nout++] = t->specials[sp].id;
      pos += strlen(t->specials[sp].s);
      continue;
    }
    size_t nxt = next_special_pos(t, s, pos, len);
    int k = encode_ordinary(t, (const unsigned char *)s + pos, nxt - pos,
                            out + nout, cap - nout);
    if (k < 0)
      return -1;
    nout += k;
    pos = nxt;
  }
  return nout;
}

int tokenizer_encode_bos(const Tokenizer *t, const char *text, int *out,
                         int cap) {
  if (!t || !out || cap < 1)
    return -1;
  out[0] = t->bos_id;
  if (!text || !text[0])
    return 1;
  int n = tokenizer_encode(t, text, out + 1, cap - 1);
  if (n < 0)
    return -1;
  return n + 1;
}
