// PLE TinyLM inference on the ESP32-S3.
//
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped. Placement follows reads-per-token rather than what happens to
// fit:
//
//   flash   PLE table + token embedding   one row of the 25.2M-parameter
//                                         table per token
//   PSRAM   staged int8 core + head, KV   read once per position
//   SRAM    scratch + norm vectors        touched many times per token
//
// The logits array stays in PSRAM: 25,353 floats is 99 KiB, and the argmax
// reads it once per token.
//
// Same llm.h that is verified against PyTorch on the host; only the platform
// hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

// int8 activations, required by the staged int8 kernel. Not bit-exact against
// the fp32 golden; verify.c must be built without this flag. Validation CE cost
// (runtime/host_verify/ppl.c, 32,768 predictions): 2.4793 -> 2.4796, ppl 11.93 / 11.94.
#define LLM_INT8_ACT 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../../runtime/llm.h"
#include "generated/vocab.h"

// Set to 1 once a display is wired up - see display.h.
// Leave 0 to run serial-only (no panel needed). Overridable at build time,
// e.g. DEFINES="-DUSE_DISPLAY=0" scripts/deploy.sh tinystories: with no panel
// attached, the I2C writes fail every token and cost wall time.
#ifndef USE_DISPLAY
#define USE_DISPLAY 1
#endif
#if USE_DISPLAY
#include "display.h"
#endif

// Story boundary: <|endoftext|> (VOCAB_EOT_ID) decodes to zero bytes, so the
// loop otherwise runs straight on into a new story. 1 stops the generation
// there; 0 keeps the fixed-length loop (upstream behaviour, benchmark mode).
// VOCAB_EOT_ID comes from tools/generate_vocab.py, which deploy.sh reruns.
#ifndef STOP_AT_EOT
#define STOP_AT_EOT 0
#endif
#if STOP_AT_EOT && !defined(VOCAB_EOT_ID)
#error "STOP_AT_EOT needs VOCAB_EOT_ID: regenerate generated/vocab.h"
#endif

// Output head on the S3 PIE int8 MAC (head_dot96_s8.S). 0 keeps the original
// scalar head; the layers use the scalar path either way.
#ifndef HEAD_SIMD
#define HEAD_SIMD 1
#endif
// With HEAD_SIMD: 1 streams the head as packed int4 (llm_head_dot96_s4, about
// half the PSRAM bytes); 0 keeps the staged int8 head (llm_head_dot96_s8).
#ifndef HEAD_INT4
#define HEAD_INT4 1
#endif

static const int PROMPT_IDS[] = {433, 447, 259, 405}; // "Once upon a time"
static const int N_GENERATE = 200;

Model model;
Scratch s;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    Serial.printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;

#if HEAD_SIMD
typedef void (*range_fn)(const QT *, const int8_t *, float, float *, int, int);
static range_fn job_fn;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    job_fn(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}
#else
static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}
#endif

static void matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
#if HEAD_SIMD
  job_fn = matvec_i8_range;
#endif
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

#if HEAD_SIMD
// ---- SIMD output head ---------------------------------------------------------
// Same arithmetic as matvec_i8_range for a one-group, 96-column tensor: the
// int32 row dots come from the PIE kernel (exact), and the float tail below is
// the same expression, so logits are bit-identical to the scalar head.
extern "C" void llm_head_dot96_s8(const int8_t *w, const int8_t *x, int32_t *out,
                                  uint32_t rows);
#define HEAD_COLS 96
#define HEAD_BLOCK 64

static void head_i8_range_simd(const QT *t, const int8_t *xq, float x_scale,
                               float *y, int row_begin, int row_end) {
  int32_t d[HEAD_BLOCK];
  for (int r0 = row_begin; r0 < row_end; r0 += HEAD_BLOCK) {
    int n = row_end - r0 < HEAD_BLOCK ? row_end - r0 : HEAD_BLOCK;
    llm_head_dot96_s8(t->w8 + (size_t)r0 * HEAD_COLS, xq, d, (uint32_t)n);
    for (int i = 0; i < n; i++) {
      float acc = 0.f;
      acc += (float)d[i] * t->scale8[r0 + i];
      y[r0 + i] = acc * x_scale;
    }
  }
}

#if HEAD_INT4
// ---- packed int4 head -------------------------------------------------------
// Restaged at boot from the model's offset-8 codes into the kernel's layout:
// per row three 16-byte chunks; chunk c byte k = element 32c+k (low nibble) and
// element 32c+16+k (high nibble), two's-complement int4. Values are unchanged,
// so the int32 dots, and with the same fp32 scales the logits, are identical.
extern "C" void llm_head_dot96_s4(const uint8_t *w, const int8_t *x, int32_t *out,
                                  uint32_t rows);
#define HEAD4_ROW_BYTES (HEAD_COLS / 2)
static const uint8_t *head4_w;     // rows * 48 B, 16-byte aligned, PSRAM
static const float *head4_sc;      // rows fp32 scales, PSRAM

static inline int head_code(const QT *t, int r, int j) {   // file code, 0..15
  uint8_t b = t->codes[(size_t)r * t->row_bytes + (j >> 1)];
  return (j & 1) ? (b >> 4) : (b & 0xF);
}

static size_t head4_bytes(const QT *t) {
  return (size_t)t->rows * HEAD4_ROW_BYTES + (size_t)t->rows * sizeof(float);
}

static void head4_stage(const QT *t, void *buffer) {
  uint8_t *w = (uint8_t *)buffer;
  float *sc = (float *)(w + (size_t)t->rows * HEAD4_ROW_BYTES);
  for (int r = 0; r < t->rows; r++) {
    uint8_t *dst = w + (size_t)r * HEAD4_ROW_BYTES;
    for (int c = 0; c < 3; c++)
      for (int k = 0; k < 16; k++) {
        int lo = head_code(t, r, 32 * c + k) - 8;
        int hi = head_code(t, r, 32 * c + 16 + k) - 8;
        dst[16 * c + k] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
      }
    sc[r] = half2float(t->scales[r]);
  }
  head4_w = w;
  head4_sc = sc;
}

static void head_i4_range_simd(const QT *t, const int8_t *xq, float x_scale,
                               float *y, int row_begin, int row_end) {
  int32_t d[HEAD_BLOCK];
  for (int r0 = row_begin; r0 < row_end; r0 += HEAD_BLOCK) {
    int n = row_end - r0 < HEAD_BLOCK ? row_end - r0 : HEAD_BLOCK;
    llm_head_dot96_s4(head4_w + (size_t)r0 * HEAD4_ROW_BYTES, xq, d, (uint32_t)n);
    for (int i = 0; i < n; i++) {
      float acc = 0.f;
      acc += (float)d[i] * head4_sc[r0 + i];
      y[r0 + i] = acc * x_scale;
    }
  }
}
#endif

static void matvec_par_head(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT] __attribute__((aligned(16)));
  float xs;
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
#if HEAD_INT4
  range_fn head_fn = head_i4_range_simd;
#else
  range_fn head_fn = head_i8_range_simd;
#endif
  job_fn = head_fn;
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  head_fn(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// Preconditions of the kernel, then every row against the scalar llm_dot_i8 on
// a fixed activation that includes both int8 extremes.
#if HEAD_INT4
// Every row of the int4 kernel against the scalar int4 arithmetic of
// matvec_q8_range ((code - 8) * x, read straight from the model's codes), on a
// fixed activation that includes both int8 extremes.
static bool head_simd_check(const QT *t) {
  if (head4_w == NULL || t->cols != HEAD_COLS || t->n_groups != 1 ||
      ((uintptr_t)head4_w & 15) != 0) {
    Serial.printf("FATAL: int4 head needs staged, 16B-aligned, %d-col, 1-group "
                  "weights (cols=%d groups=%d w=%p)\n",
                  HEAD_COLS, t->cols, t->n_groups, head4_w);
    return false;
  }
  static int8_t xt[HEAD_COLS] __attribute__((aligned(16)));
  for (int j = 0; j < HEAD_COLS; j++)
    xt[j] = (j % 3 == 0) ? 127 : (j % 3 == 1) ? -127 : (int8_t)(j * 37 - 90);
  int32_t d[HEAD_BLOCK];
  int bad = 0;
  for (int r0 = 0; r0 < t->rows; r0 += HEAD_BLOCK) {
    int n = t->rows - r0 < HEAD_BLOCK ? t->rows - r0 : HEAD_BLOCK;
    llm_head_dot96_s4(head4_w + (size_t)r0 * HEAD4_ROW_BYTES, xt, d, (uint32_t)n);
    for (int i = 0; i < n; i++) {
      int32_t ref = 0;
      for (int j = 0; j < HEAD_COLS; j++) ref += (head_code(t, r0 + i, j) - 8) * (int)xt[j];
      if (d[i] != ref) {
        if (bad < 4) Serial.printf("simd head mismatch row %d: %ld vs %ld\n",
                                   r0 + i, (long)d[i], (long)ref);
        bad++;
      }
    }
  }
  Serial.printf("simd head selftest: %d/%d rows %s\n", t->rows - bad, t->rows,
                bad ? "FAIL" : "ok");
  return bad == 0;
}
#else
static bool head_simd_check(const QT *t) {
  if (t->w8 == NULL || t->cols != HEAD_COLS || t->n_groups != 1 ||
      ((uintptr_t)t->w8 & 15) != 0) {
    Serial.printf("FATAL: SIMD head needs staged, 16B-aligned, %d-col, 1-group "
                  "weights (cols=%d groups=%d w8=%p)\n",
                  HEAD_COLS, t->cols, t->n_groups, t->w8);
    return false;
  }
  static int8_t xt[HEAD_COLS] __attribute__((aligned(16)));
  for (int j = 0; j < HEAD_COLS; j++)
    xt[j] = (j % 3 == 0) ? 127 : (j % 3 == 1) ? -127 : (int8_t)(j * 37 - 90);
  int32_t d[HEAD_BLOCK];
  int bad = 0;
  for (int r0 = 0; r0 < t->rows; r0 += HEAD_BLOCK) {
    int n = t->rows - r0 < HEAD_BLOCK ? t->rows - r0 : HEAD_BLOCK;
    llm_head_dot96_s8(t->w8 + (size_t)r0 * HEAD_COLS, xt, d, (uint32_t)n);
    for (int i = 0; i < n; i++)
      if (d[i] != llm_dot_i8(t->w8 + (size_t)(r0 + i) * HEAD_COLS, xt, HEAD_COLS)) {
        if (bad < 4) Serial.printf("simd head mismatch row %d\n", r0 + i);
        bad++;
      }
  }
  Serial.printf("simd head selftest: %d/%d rows %s\n", t->rows - bad, t->rows,
                bad ? "FAIL" : "ok");
  return bad == 0;
}
#endif  // HEAD_INT4
#endif

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  Serial.printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// Emit one token to every active output (serial always; panel when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills.
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM ===");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads,
                c->ffn, c->ple_dim, part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#endif

  // The model header states how many logits it produces; vocab.h carries the
  // decode table. If they disagree, every emitted token would be decoded
  // against the wrong table.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = llm_stage_core_int8_alloc(&model, ps);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it too: it is 85%% of the dense MACs.
  {
#if HEAD_SIMD && HEAD_INT4
    // Packed int4 + fp32 scales, 16-byte aligned for ee.vld.128. out_head keeps
    // w8 == NULL, so any non-hook caller still takes the exact int4 path.
    size_t hb = head4_bytes(&model.out_head);
    void *b = heap_caps_aligned_alloc(16, hb, MALLOC_CAP_SPIRAM);
    if (!b) {
      Serial.printf("FATAL: required PSRAM allocation failed: int4 head (%u bytes)\n",
                    (unsigned)hb);
      while (1) delay(1000);
    }
    psram_used += hb;
    head4_stage(&model.out_head, b);
    Serial.printf("head   -> PSRAM  packed int4, %u B\n", (unsigned)hb);
#elif HEAD_SIMD
    // 16-byte aligned for ee.vld.128.
    size_t hb = llm_stage_int8_bytes(&model.out_head);
    void *b = heap_caps_aligned_alloc(16, hb, MALLOC_CAP_SPIRAM);
    if (!b) {
      Serial.printf("FATAL: required PSRAM allocation failed: staged head (%u bytes)\n",
                    (unsigned)hb);
      while (1) delay(1000);
    }
    psram_used += hb;
#else
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
#endif
#if !(HEAD_SIMD && HEAD_INT4)
    llm_stage_int8(&model.out_head, b);
#endif
    ++staged;
  }
  Serial.printf("weights-> PSRAM  %d tensors int8, %.2f MB allocated\n",
                staged, psram_used / 1048576.0);

  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = matvec_par;
#if HEAD_SIMD
    if (!head_simd_check(&model.out_head)) while (1) delay(1000);
    model.head_matvec  = matvec_par_head;
#if HEAD_INT4
    Serial.println("head   -> PIE int4 (head_dot96_s4)");
#else
    Serial.println("head   -> PIE int8 (head_dot96_s8)");
#endif
#else
    model.head_matvec  = matvec_par;
#endif
  } else {
    Serial.println("dual-core worker failed; running single core");
  }

  // FNV-1a over the mapped image. scripts/deploy.sh prints the same value for
  // the file it flashed; the two must agree.
  {
    const uint8_t *img = (const uint8_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: bytes=%u fp=%08x sram=%uB psram=%.2fMB\n",
                  (unsigned)model.image_bytes, fp,
                  (unsigned)(sram_used + STATIC_SRAM_BYTES),
                  psram_used / 1048576.0);
  }
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

  // ---- generate ----
  Serial.print(">>> ");
  int n_prompt = sizeof(PROMPT_IDS) / sizeof(int);
  int pos = 0, tok = 0;
  int64_t decode_us = 0;
  int decoded = 0;

  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = PROMPT_IDS[i];
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);

  int64_t t_start = esp_timer_get_time();
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < model.out_vocab; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    tok = best;
#if STOP_AT_EOT
    if (tok == VOCAB_EOT_ID) break;  // end of story: stop instead of starting another
#endif
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  if (decoded)  // zero when the first generated token is the boundary
    Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                  decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#if USE_DISPLAY
  if (decoded) display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
  blink(0);
}

void loop() { delay(10000); }
