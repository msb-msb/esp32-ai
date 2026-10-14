// ia_s3_port.h - insiderllm-p4 (Miu, 2026-10-11). The S3 firmware's interactive line protocol,
// PREFILL_SKIP, splitmix64 PRNG, ia_sample and FAST_SAMPLE, copied VERBATIM from our S3 board
// build (esp32_tinystories.ino sha256 ea055c359ade863d7d291faea1745825f226d2193d3d88fdb39662edc05c7f63,
// lines 106-174, 819-836, 852-1350) by p4/tools/extract_s3_interactive.sh. Do not edit by hand:
// re-run the script. Included by the P4 sketch, which builds on esp32-ai PR #24 (Barkin Sarikartal).
// Requires from the sketch: model, s, blink(), llm_profile_reset(), and with LOGIT_CHECKSUM
// logit_hash[] / fnv1a64() / LOGIT_CHECKSUM_TOKENS; VOCAB_EOT_ID from generated/vocab.h.
#ifndef IA_S3_PORT_H
#define IA_S3_PORT_H
// Interactive serial mode (tools/chat.py). 0 keeps the fixed benchmark run in
// setup(). 1 skips it and serves a line protocol on the native USB serial:
//   host -> board  "P <id> <id> ...\n"                     prompt token ids
//                  "G n=<N> t=<temp> p=<top_p> s=<seed> eot=<0|1>\n"
//                  "?\n"                                    ready check
//   board -> host  "T <id>"  per generated token, then
//                  "D ttft_us=<..> decode_ms_per_tok=<..> n=<..>"
//                  "R ready ..." at boot and on "?", "E <reason>" on a bad line
// Each G runs the stored prompt from position 0 (one-shot, no multi-turn
// state). t=0 is the benchmark's argmax loop; t>0 samples with temperature and
// top-p from a PRNG seeded by s. TTFT is G line received -> first T line,
// decode_ms_per_tok is llm_forward time per generated token as in the
// benchmark's throughput line.
#ifndef INTERACTIVE
#define INTERACTIVE 0
#endif
#if INTERACTIVE && !defined(VOCAB_EOT_ID)
#error "INTERACTIVE needs VOCAB_EOT_ID: regenerate generated/vocab.h"
#endif

// Prompt positions before the last run every layer (their K/V are needed) but
// not the output head: nothing reads those logits. 1 skips it; 0 computes it.
// The last prompt position always computes logits, so no value changes.
#ifndef PREFILL_SKIP
#define PREFILL_SKIP 0
#endif

// Sampling-path profile: SAMPLE_PROFILE=1 counts CPU cycles per stage of each
// t>0 draw and prints them per sampled token on an "S" line after D. The
// profiled ia_sample splits the fused scale/exp/sum loop into three passes;
// each value and the order of the sum are the same, so the draws are too.
// 0 = the original ia_sample, no timing code.
#ifndef SAMPLE_PROFILE
#define SAMPLE_PROFILE 0
#endif

// Fast sampling for t>0, top_p<1 (FAST_SAMPLE=1; 0 = ia_sample only). One
// fused pass after the max: tokens with (l - max)/t < -24 are dropped (their
// total probability is < 25,352 e^-24 = 9.57e-7); the rest go into the
// normaliser through a fast exp (within 1 ulp of expf); the few with weight
// >= 2^-10 are recomputed with expf and are the only candidates for top-p,
// selected with a heap, and for the draw, which is ia_sample's on those exact
// weights. If the candidates overflow or cannot reach top_p, ia_sample runs on
// the untouched logits. t=0 never comes here.
#ifndef FAST_SAMPLE
#define FAST_SAMPLE 0
#endif
#if FAST_SAMPLE && !INTERACTIVE
#error "FAST_SAMPLE is the INTERACTIVE sampler"
#endif

// Power-logger sync (tools/power-logger): SYNC_GPIO=<pin> drives that pin high
// for each G command, from just before the prompt's prefill to just after the
// last token, and low otherwise; the logger measures energy between the edges.
// Both edges sit outside the timed llm_forward calls, so decode_ms_per_tok is
// unchanged. The pin is a push-pull output driven low from the start of
// setup(), so it is always driven: left as an input, it fought the logger's
// pull-down and sat at ~0.95 V, where the logger's input chattered.
// -1 (default) compiles all of it out. Share a ground with the logger.
#ifndef SYNC_GPIO
#define SYNC_GPIO -1
#endif
#if SYNC_GPIO >= 0
#define SYNC_INIT() do { pinMode(SYNC_GPIO, OUTPUT); digitalWrite(SYNC_GPIO, LOW); } while (0)
#define SYNC_SET(v) digitalWrite(SYNC_GPIO, (v) ? HIGH : LOW)
#else
#define SYNC_INIT() do { } while (0)
#define SYNC_SET(v) do { } while (0)
#endif

// One prompt position. With PREFILL_SKIP and !last, the head hook is swapped for
// a no-op for this call only (llm_forward still applies out_norm to its own
// scratch x, which the next call overwrites).
#if PREFILL_SKIP
static void head_skip(const QT *, const float *, float *) {}
#endif
static void forward_prompt(int tok, int pos, bool last) {
#if PREFILL_SKIP
  if (!last) {
    void (*hm)(const QT *, const float *, float *) = model.head_matvec;
    model.head_matvec = head_skip;
    llm_forward(&model, tok, pos, &s);
    model.head_matvec = hm;
    return;
  }
#endif
  llm_forward(&model, tok, pos, &s);
}

#if INTERACTIVE
// ---- interactive serial mode ---------------------------------------------------
// Prime the KV cache with ids[0..n), positions *pos onward. The logits of the
// last position are the first generated token's.
static void prefill(const int *ids, int n, int *pos) {
  for (int i = 0; i < n; i++)
    forward_prompt(ids[i], (*pos)++, i == n - 1);
}

#define IA_LINE_MAX 4096
#define IA_CAND_MAX 1024
static bool ia_ready = false;
static int ia_prompt[1024];
static int ia_n_prompt = 0;
static char ia_line[IA_LINE_MAX];
static int ia_len = 0;
static bool ia_overflow = false;
static uint64_t ia_rng;
static float ia_cand[IA_CAND_MAX];

// Zero the KV cache. Attention only reads positions 0..pos, which each run
// rewrites before reading, so this changes no value; it keeps one run's state
// from ever reaching the next. Runs on P and after each D, outside any TTFT.
static void ia_reset_kv() {
  size_t n = (size_t)model.c.n_layers * model.c.seq_len * model.c.dim * sizeof(float);
  memset(s.kcache, 0, n);
  memset(s.vcache, 0, n);
}

static void ia_ready_line() {
  Serial.printf("R ready vocab=%d seq_len=%d checksum=%d prefill_skip=%d sample_profile=%d "
                "fast_sample=%d\n", model.out_vocab, model.c.seq_len, LOGIT_CHECKSUM,
                PREFILL_SKIP, SAMPLE_PROFILE, FAST_SAMPLE);
}

// splitmix64: small, deterministic, seeded straight from s=.
static uint64_t ia_next() {
  uint64_t z = (ia_rng += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static float ia_uniform() {   // [0, 1), 24 bits
  return (float)(ia_next() >> 40) * (1.0f / 16777216.0f);
}

static int ia_cmp_desc(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return (x < y) - (x > y);
}

#if FAST_SAMPLE
static uint32_t fs_fallbacks;   // t>0 tokens that went through ia_sample instead
#endif
#if SAMPLE_PROFILE
#include "esp_cpu.h"
enum { SP_MAX, SP_SCALE, SP_EXP, SP_NORM, SP_SELECT, SP_CUMSUM, SP_DRAW, SP_TOTAL, SP_N };
static uint64_t sp_cyc[SP_N];
static uint32_t sp_tokens, sp_passes, sp_bisect, sp_surv_max;
static uint64_t sp_cand, sp_surv, sp_nucleus;
#if FAST_SAMPLE
enum { FP_MAX, FP_FUSED, FP_SELECT, FP_DRAW, FP_TOTAL, FP_N };
static uint64_t fp_cyc[FP_N], fp_cand, fp_nucleus;
static uint32_t fp_tokens;
#endif
#define SP_T(x) uint32_t x = esp_cpu_get_cycle_count()
#define SP_ADD(i, a, b) (sp_cyc[i] += (uint32_t)((b) - (a)))
static void sp_reset() {
#if FAST_SAMPLE
  memset(fp_cyc, 0, sizeof(fp_cyc));
  fp_cand = fp_nucleus = 0;
  fp_tokens = 0;
#endif
  memset(sp_cyc, 0, sizeof(sp_cyc));
  sp_tokens = sp_passes = sp_bisect = sp_surv_max = 0;
  sp_cand = sp_surv = sp_nucleus = 0;
}
static void sp_report() {
#if FAST_SAMPLE
  // Fast-path tokens only; fallback tokens are counted, and their time is in
  // the S line's ia_sample fields only if every token fell back.
  if (fp_tokens) {
    double kc = (double)fp_tokens * getCpuFrequencyMhz() * 1000.0;
    Serial.printf("S fast=1 tok=%u max=%.4f fused=%.4f select=%.4f draw=%.4f total=%.4f "
                  "fallback=%u cand=%.1f nucleus=%.1f\n", fp_tokens,
                  fp_cyc[FP_MAX] / kc, fp_cyc[FP_FUSED] / kc, fp_cyc[FP_SELECT] / kc,
                  fp_cyc[FP_DRAW] / kc, fp_cyc[FP_TOTAL] / kc, fs_fallbacks,
                  (double)fp_cand / fp_tokens, (double)fp_nucleus / fp_tokens);
    return;
  }
#endif
  if (!sp_tokens) return;
  double k = (double)sp_tokens * getCpuFrequencyMhz() * 1000.0;   // cycles per ms * tokens
  Serial.printf("S tok=%u max=%.4f scale=%.4f exp=%.4f norm=%.4f select=%.4f cumsum=%.4f "
                "draw=%.4f total=%.4f passes=%.2f cand=%.1f bisect=%u surv24=%.1f "
                "surv24_max=%u nucleus=%.1f\n", sp_tokens,
                sp_cyc[SP_MAX] / k, sp_cyc[SP_SCALE] / k, sp_cyc[SP_EXP] / k,
                sp_cyc[SP_NORM] / k, sp_cyc[SP_SELECT] / k, sp_cyc[SP_CUMSUM] / k,
                sp_cyc[SP_DRAW] / k, sp_cyc[SP_TOTAL] / k,
                (double)sp_passes / sp_tokens, (double)sp_cand / sp_tokens, sp_bisect,
                (double)sp_surv / sp_tokens, sp_surv_max, (double)sp_nucleus / sp_tokens);
}
#else
#define SP_T(x)
#define SP_ADD(i, a, b)
#endif

// Sum and count of the weights >= c.
static float ia_mass(const float *p, int V, float c, int *count) {
  float m = 0.f;
  int k = 0;
  for (int v = 0; v < V; v++)
    if (p[v] >= c) { m += p[v]; k++; }
  if (count) *count = k;
  return m;
}

// Temperature + top-p over the logits (overwritten with the unnormalized
// weights exp((l - max) / t)). The nucleus is every token whose weight is at
// least tau, the smallest weight of the top tokens that reach top_p of the
// total; the draw walks it in id order, so no full sort is needed.
#if SAMPLE_PROFILE
// Profiled copy of the function below. Untimed extras: the count of tokens with
// (l - max)/t >= -24 and the nucleus size.
static int ia_sample(float *l, int V, float temp, float top_p) {
  SP_T(t0);
  float mx = l[0];
  for (int v = 1; v < V; v++) if (l[v] > mx) mx = l[v];
  SP_T(t1);
  float inv = 1.f / temp, Z = 0.f;
  for (int v = 0; v < V; v++) l[v] = (l[v] - mx) * inv;
  SP_T(t2);
  {
    uint32_t k = 0;
    for (int v = 0; v < V; v++) if (l[v] >= -24.0f) k++;
    sp_surv += k;
    if (k > sp_surv_max) sp_surv_max = k;
  }
  SP_T(t2b);
  for (int v = 0; v < V; v++) l[v] = expf(l[v]);
  SP_T(t3);
  for (int v = 0; v < V; v++) Z += l[v];
  SP_T(t4);
  uint32_t t5 = t4, t6 = t4;
  float tau = 0.f;
  if (top_p < 1.f) {
    float target = top_p * Z, c = 0.5f;
    int cnt = 0;
    sp_passes++;
    while (ia_mass(l, V, c, &cnt) < target && c > 1e-30f) { c *= 0.0625f; sp_passes++; }
    if (cnt <= IA_CAND_MAX) {
      int k = 0;
      for (int v = 0; v < V; v++) if (l[v] >= c) ia_cand[k++] = l[v];
      qsort(ia_cand, k, sizeof(float), ia_cmp_desc);
      sp_cand += k;
      t5 = esp_cpu_get_cycle_count();
      float acc = 0.f;
      tau = ia_cand[k - 1];
      for (int i = 0; i < k; i++) {
        acc += ia_cand[i];
        if (acc >= target) { tau = ia_cand[i]; break; }
      }
      t6 = esp_cpu_get_cycle_count();
    } else {
      float lo = c, hi = c * 16.f;
      for (int it = 0; it < 32; it++) {
        float mid = 0.5f * (lo + hi);
        if (ia_mass(l, V, mid, NULL) >= target) lo = mid; else hi = mid;
      }
      tau = lo;
      sp_bisect++;
      t5 = t6 = esp_cpu_get_cycle_count();
    }
  }
  float tot = ia_mass(l, V, tau, NULL);
  float r = ia_uniform() * tot, acc = 0.f;
  int last = 0, pick = -1;
  for (int v = 0; v < V; v++)
    if (l[v] >= tau) {
      acc += l[v];
      last = v;
      if (acc > r) { pick = v; break; }
    }
  if (pick < 0) pick = last;
  SP_T(t7);
  {
    int nn = 0;
    ia_mass(l, V, tau, &nn);
    sp_nucleus += nn;
  }
  SP_ADD(SP_MAX, t0, t1); SP_ADD(SP_SCALE, t1, t2); SP_ADD(SP_EXP, t2b, t3);
  SP_ADD(SP_NORM, t3, t4); SP_ADD(SP_SELECT, t4, t5); SP_ADD(SP_CUMSUM, t5, t6);
  SP_ADD(SP_DRAW, t6, t7);
  sp_cyc[SP_TOTAL] += (uint32_t)(t7 - t0) - (uint32_t)(t2b - t2);
  sp_tokens++;
  return pick;
}
#else
static int ia_sample(float *l, int V, float temp, float top_p) {
  float mx = l[0];
  for (int v = 1; v < V; v++) if (l[v] > mx) mx = l[v];
  float inv = 1.f / temp, Z = 0.f;
  for (int v = 0; v < V; v++) { l[v] = expf((l[v] - mx) * inv); Z += l[v]; }
  float tau = 0.f;
  if (top_p < 1.f) {
    float target = top_p * Z, c = 0.5f;
    int cnt = 0;
    // Lower the cutoff 16x at a time until the weights above it reach target.
    while (ia_mass(l, V, c, &cnt) < target && c > 1e-30f) c *= 0.0625f;
    if (cnt <= IA_CAND_MAX) {
      int k = 0;
      for (int v = 0; v < V; v++) if (l[v] >= c) ia_cand[k++] = l[v];
      qsort(ia_cand, k, sizeof(float), ia_cmp_desc);
      float acc = 0.f;
      tau = ia_cand[k - 1];
      for (int i = 0; i < k; i++) {
        acc += ia_cand[i];
        if (acc >= target) { tau = ia_cand[i]; break; }
      }
    } else {
      // Flat distribution: bisect for the largest cutoff that still reaches it.
      float lo = c, hi = c * 16.f;
      for (int it = 0; it < 32; it++) {
        float mid = 0.5f * (lo + hi);
        if (ia_mass(l, V, mid, NULL) >= target) lo = mid; else hi = mid;
      }
      tau = lo;
    }
  }
  float tot = ia_mass(l, V, tau, NULL);
  float r = ia_uniform() * tot, acc = 0.f;
  int last = 0;
  for (int v = 0; v < V; v++)
    if (l[v] >= tau) {
      acc += l[v];
      last = v;
      if (acc > r) return v;
    }
  return last;
}
#endif  // SAMPLE_PROFILE

#if FAST_SAMPLE
#define FS_C 24.0f                 // 25,352 * e^-24 = 9.57e-7 < 1e-6
#define FS_G 0.0009765625f         // 2^-10: weights recomputed with expf
#define FS_CAND_MAX 1024
static float fs_w[FS_CAND_MAX], fs_heap[FS_CAND_MAX];
static uint16_t fs_id[FS_CAND_MAX];

// exp(s) for s in [-24, 0]: Cody-Waite reduction s = k ln2 + r, |r| <= ln2/2,
// degree-6 near-minimax polynomial for exp(r), times 2^k from the exponent
// bits. Host check over all 1,103,101,953 floats in [-24, 0]: within 1 ulp of
// glibc expf everywhere (99.25% identical), max rel error 1.04e-7. The boot
// self-test repeats the comparison against this toolchain's expf.
static inline float fs_exp(float s) {
  float kf = (float)(int)(s * 1.44269504f - 0.5f);   // round(s / ln2), s <= 0
  float r = s - kf * 0.693145751953125f;              // ln2 high part, exact product
  r = r - kf * 1.428606765330187e-06f;                // ln2 low part
  float p = 0.001351682236418128f;
  p = p * r + 0.008370536379516125f;
  p = p * r + 0.04167335852980614f;
  p = p * r + 0.16666476428508759f;
  p = p * r + 0.4999997317790985f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  union { uint32_t u; float f; } b;
  b.u = (uint32_t)((int)kf + 127) << 23;
  return p * b.f;
}

// Every 4,099th float in [-24, 0] (about 269k values) against expf.
static bool fs_exp_check() {
  uint32_t hi; float m24 = -24.0f; memcpy(&hi, &m24, 4);
  uint32_t n = 0, same = 0, maxulp = 0;
  for (uint32_t u = 0x80000000u; u <= hi; u += 4099) {
    float x; memcpy(&x, &u, 4);
    float a = fs_exp(x), e = expf(x);
    int32_t ia, ie; memcpy(&ia, &a, 4); memcpy(&ie, &e, 4);
    uint32_t d = ia > ie ? ia - ie : ie - ia;
    if (d == 0) same++;
    if (d > maxulp) maxulp = d;
    n++;
  }
  Serial.printf("fast exp selftest: %u values, %.2f%% identical to expf, max %u ulp %s\n",
                n, 100.0 * same / n, maxulp, maxulp <= 1 ? "ok" : "FAIL");
  return maxulp <= 1;
}

static void fs_sift(float *h, int n, int i) {   // max-heap
  for (;;) {
    int a = 2 * i + 1, b = a + 1, m = i;
    if (a < n && h[a] > h[m]) m = a;
    if (b < n && h[b] > h[m]) m = b;
    if (m == i) return;
    float tmp = h[i]; h[i] = h[m]; h[m] = tmp;
    i = m;
  }
}

static int fs_sample(float *l, int V, float temp, float top_p) {
  if (!(top_p < 1.f)) { fs_fallbacks++; return ia_sample(l, V, temp, top_p); }
  SP_T(t0);
  float mx = l[0];
  for (int v = 1; v < V; v++) if (l[v] > mx) mx = l[v];
  SP_T(t1);
  float inv = 1.f / temp, Z = 0.f;
  int k = 0;
  for (int v = 0; v < V; v++) {
    float sc = (l[v] - mx) * inv;
    if (sc >= -FS_C) {
      float w = fs_exp(sc);
      if (w >= FS_G) {
        if (k == FS_CAND_MAX) { fs_fallbacks++; return ia_sample(l, V, temp, top_p); }
        w = expf(sc);
        fs_w[k] = w; fs_id[k] = (uint16_t)v; k++;
      }
      Z += w;
    }
  }
  SP_T(t2);
  // Top-p: pop candidates largest first until the running sum reaches target.
  float target = top_p * Z, acc = 0.f, tau = -1.f;
  memcpy(fs_heap, fs_w, k * sizeof(float));
  for (int i = k / 2 - 1; i >= 0; i--) fs_sift(fs_heap, k, i);
  for (int n = k; n > 0;) {
    float top = fs_heap[0];
    acc += top;
    if (acc >= target) { tau = top; break; }
    fs_heap[0] = fs_heap[--n];
    fs_sift(fs_heap, n, 0);
  }
  if (tau < 0.f) { fs_fallbacks++; return ia_sample(l, V, temp, top_p); }
  SP_T(t3);
  // ia_sample's draw over the nucleus (candidates are in id order).
  float tot = 0.f;
  for (int i = 0; i < k; i++) if (fs_w[i] >= tau) tot += fs_w[i];
  float r = ia_uniform() * tot;
  int last = 0, pick = -1;
  acc = 0.f;
  for (int i = 0; i < k; i++)
    if (fs_w[i] >= tau) {
      acc += fs_w[i];
      last = fs_id[i];
      if (acc > r) { pick = fs_id[i]; break; }
    }
  if (pick < 0) pick = last;
#if SAMPLE_PROFILE
  SP_T(t4);
  fp_cyc[FP_MAX] += (uint32_t)(t1 - t0); fp_cyc[FP_FUSED] += (uint32_t)(t2 - t1);
  fp_cyc[FP_SELECT] += (uint32_t)(t3 - t2); fp_cyc[FP_DRAW] += (uint32_t)(t4 - t3);
  fp_cyc[FP_TOTAL] += (uint32_t)(t4 - t0);
  fp_cand += k;
  for (int i = 0; i < k; i++) if (fs_w[i] >= tau) fp_nucleus++;
  fp_tokens++;
#endif
  return pick;
}
#endif  // FAST_SAMPLE

static void ia_cmd_prompt(char *p) {
  int n = 0;
  char *end;
  for (;;) {
    while (*p == ' ') p++;
    if (!*p) break;
    long id = strtol(p, &end, 10);
    if (end == p || (*end && *end != ' ')) { Serial.println("E bad prompt id"); ia_n_prompt = 0; return; }
    if (id < 0 || id >= model.out_vocab) {
      Serial.printf("E prompt id %ld out of range 0..%d\n", id, model.out_vocab - 1);
      ia_n_prompt = 0; return;
    }
    if (n >= model.c.seq_len || n >= (int)(sizeof(ia_prompt) / sizeof(ia_prompt[0]))) {
      Serial.printf("E prompt longer than seq_len %d\n", model.c.seq_len);
      ia_n_prompt = 0; return;
    }
    ia_prompt[n++] = (int)id;
    p = end;
  }
  if (n == 0) { Serial.println("E empty prompt"); ia_n_prompt = 0; return; }
  ia_n_prompt = n;
  ia_reset_kv();
}

static void ia_cmd_generate(char *p, int64_t t_g) {
  long n = 200, eot = 0;
  float temp = 0.f, top_p = 1.f;
  unsigned long long seed = 0;
  for (char *kv = strtok(p, " "); kv; kv = strtok(NULL, " ")) {
    char *eq = strchr(kv, '=');
    if (!eq) { Serial.printf("E bad field %s\n", kv); return; }
    *eq = 0;
    const char *v = eq + 1;
    if (!strcmp(kv, "n")) n = strtol(v, NULL, 10);
    else if (!strcmp(kv, "t")) temp = strtof(v, NULL);
    else if (!strcmp(kv, "p")) top_p = strtof(v, NULL);
    else if (!strcmp(kv, "s")) seed = strtoull(v, NULL, 10);
    else if (!strcmp(kv, "eot")) eot = strtol(v, NULL, 10);
    else { Serial.printf("E unknown field %s\n", kv); return; }
  }
  if (ia_n_prompt == 0) { Serial.println("E no prompt (send P first)"); return; }
  if (n < 1) { Serial.println("E n must be >= 1"); return; }
  if (ia_n_prompt + n > model.c.seq_len) {
    Serial.printf("E prompt %d + n %ld > seq_len %d\n", ia_n_prompt, n, model.c.seq_len);
    return;
  }
  if (!(temp >= 0.f) || !(top_p > 0.f && top_p <= 1.f)) {
    Serial.println("E need t >= 0 and 0 < p <= 1"); return;
  }
  ia_rng = seed;
#if FAST_SAMPLE
  fs_fallbacks = 0;
#endif
#if SAMPLE_PROFILE
  sp_reset();
#endif

  int pos = 0, tok = 0, decoded = 0;
  int64_t decode_us = 0, ttft_us = -1;
  SYNC_SET(1);
  prefill(ia_prompt, ia_n_prompt, &pos);
  llm_profile_reset(&s);
  for (int step = 0; step < n && pos < model.c.seq_len; step++) {
#if LOGIT_CHECKSUM
    if (step < LOGIT_CHECKSUM_TOKENS)
      logit_hash[step] = fnv1a64(s.logits, (size_t)model.out_vocab * sizeof(float),
                                 14695981039346656037ULL);
#endif
    if (temp == 0.f) {   // the benchmark's argmax, unchanged
      int best = 0; float bv = -1e30f;
      for (int v = 0; v < model.out_vocab; v++)
        if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
      tok = best;
    } else {
#if FAST_SAMPLE
      tok = fs_sample(s.logits, model.out_vocab, temp, top_p);
#else
      tok = ia_sample(s.logits, model.out_vocab, temp, top_p);
#endif
    }
    if (eot && tok == VOCAB_EOT_ID) break;
    if (ttft_us < 0) ttft_us = esp_timer_get_time() - t_g;
    Serial.printf("T %d\n", tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens
  }
  SYNC_SET(0);
  Serial.printf("D ttft_us=%lld decode_ms_per_tok=%.3f n=%d\n", (long long)ttft_us,
                decoded ? decode_us / 1000.0 / decoded : 0.0, decoded);
#if LOGIT_CHECKSUM
  {
    int k = decoded < LOGIT_CHECKSUM_TOKENS ? decoded : LOGIT_CHECKSUM_TOKENS;
    for (int i = 0; i < k; i++)
      Serial.printf("H %02d %016llx\n", i, (unsigned long long)logit_hash[i]);
    Serial.printf("H all %016llx n=%d\n",
                  (unsigned long long)fnv1a64(logit_hash, k * sizeof(uint64_t),
                                              14695981039346656037ULL), k);
  }
#endif
#if SAMPLE_PROFILE
  sp_report();
#endif
  blink(0);
  ia_reset_kv();
}

static void ia_handle(char *line, int64_t t_rx) {
  int n = strlen(line);
  while (n && (line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
  if (n == 0) return;
  if (!ia_ready) { Serial.println("E not ready (boot failed)"); return; }
  if (line[0] == 'P' && (line[1] == ' ' || !line[1])) ia_cmd_prompt(line + 1);
  else if (line[0] == 'G' && (line[1] == ' ' || !line[1])) ia_cmd_generate(line + 1, t_rx);
  else if (line[0] == '?' && !line[1]) ia_ready_line();
  else Serial.println("E unknown command");
}

static void ia_poll() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;
    if (c == '\n') {
      int64_t t = esp_timer_get_time();
      ia_line[ia_len] = 0;
      if (ia_overflow) Serial.printf("E line longer than %d bytes\n", IA_LINE_MAX - 1);
      else ia_handle(ia_line, t);
      ia_len = 0; ia_overflow = false;
    } else if (ia_len < IA_LINE_MAX - 1) {
      ia_line[ia_len++] = (char)c;
    } else {
      ia_overflow = true;
    }
  }
}
#endif  // INTERACTIVE
#endif  // IA_S3_PORT_H
