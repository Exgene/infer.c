#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
  char **id_to_token;
  int vocab_size;
} Tokenizer;

int tokenizer_load(const char *path, int vocab_size, Tokenizer *t);
void tokenizer_free(Tokenizer *t);

/* Reverse lookup: id → raw vocab piece (byte-level alphabet). "" if missing. */
const char *tokenizer_lookup(const Tokenizer *t, int id);

/* Decode one id to UTF-8 bytes. Returns byte count, or 0 if skipped/missing.
   Special tokens (<|...|>) are skipped. out is not always NUL-terminated
   beyond the returned count; a NUL is written if cap allows. */
int tokenizer_decode_id(const Tokenizer *t, int id, char *out, int cap);

#endif
