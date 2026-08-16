#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
  char *key;
  int val;
} TokSlot;

typedef struct {
  TokSlot *slots;
  int cap;
  int len;
} TokMap;

typedef struct {
  const char *s;
  int id;
} TokSpecial;

typedef struct {
  char **id_to_token;
  int vocab_size;
  TokMap piece_to_id;
  TokMap merge_rank;
  TokSpecial *specials;
  int n_specials;
  int bos_id;
} Tokenizer;

int tokenizer_load(const char *path, int vocab_size, Tokenizer *t);
void tokenizer_free(Tokenizer *t);
const char *tokenizer_lookup(const Tokenizer *t, int id);
int tokenizer_decode_id(const Tokenizer *t, int id, char *out, int cap);
int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int cap);
int tokenizer_encode_bos(const Tokenizer *t, const char *text, int *out,
                         int cap);

#endif
