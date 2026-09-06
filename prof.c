#include "prof.h"

#include <stdio.h>
#include <time.h>

ProfTime prof_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  ProfTime t;
  t.sec = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
  return t;
}

double prof_elapsed(ProfTime start, ProfTime end) {
  return end.sec - start.sec;
}

void prof_report(double load_sec, int prompt_tokens, double prefill_sec,
                 int gen_tokens, double decode_sec, int omp_threads) {
  double pp_tps =
      (prompt_tokens > 0 && prefill_sec > 0.0) ? prompt_tokens / prefill_sec : 0.0;
  double tg_tps =
      (gen_tokens > 0 && decode_sec > 0.0) ? gen_tokens / decode_sec : 0.0;
  double infer_sec = prefill_sec + decode_sec;

  fprintf(stderr, "\n--- profile ---\n");
  fprintf(stderr, "threads:       %d\n", omp_threads);
  fprintf(stderr, "load:          %.3f s\n", load_sec);
  fprintf(stderr, "prefill:       %.3f s  (%d tokens, %.2f tok/s)\n",
          prefill_sec, prompt_tokens, pp_tps);
  fprintf(stderr, "decode:        %.3f s  (%d tokens, %.2f tok/s)\n", decode_sec,
          gen_tokens, tg_tps);
  fprintf(stderr, "inference:     %.3f s  (prefill + decode)\n", infer_sec);
  if (prompt_tokens + gen_tokens > 0 && infer_sec > 0.0) {
    fprintf(stderr, "overall:       %.2f tok/s  (%d total tokens)\n",
            (prompt_tokens + gen_tokens) / infer_sec,
            prompt_tokens + gen_tokens);
  }
}
