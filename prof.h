#ifndef PROF_H
#define PROF_H

#include <stdint.h>

typedef struct {
  double sec;
} ProfTime;

typedef struct {
  double total_sec;
  uint64_t count;
} ProfStat;

ProfTime prof_now(void);
double prof_elapsed(ProfTime start, ProfTime end);

void prof_report(double load_sec, int prompt_tokens, double prefill_sec,
                 int gen_tokens, double decode_sec, int omp_threads);

#endif
