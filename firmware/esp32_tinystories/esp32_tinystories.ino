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
// Core (non-head) int8 matvecs on PIE: the int32 row dots only; quantization,
// the dual-core split and every float op are unchanged. 0 keeps matvec_par.
#ifndef CORE_SIMD
#define CORE_SIMD 1
#endif
#if CORE_SIMD && !HEAD_SIMD
#error "CORE_SIMD uses the HEAD_SIMD worker plumbing and kernels"
#endif
// 1 puts the logits buffer (out_vocab floats, 99 KiB) in internal SRAM, so the
// head's writes and the argmax read stay off PSRAM; 0 keeps it in PSRAM.
// Values are unchanged.
#ifndef LOGITS_SRAM
#define LOGITS_SRAM 1
#endif
// With CORE_SIMD: 1 streams the core matvec weights as packed int4 in the head's
// kernel layout (llm_head_dot96_s4 for 96 columns and for down zero-padded to
// 96, llm_dot128_s4 for 128); 0 keeps the staged int8 kernels.
#ifndef CORE_INT4
#define CORE_INT4 1
#endif
#if CORE_INT4 && !(CORE_SIMD && HEAD_INT4)
#error "CORE_INT4 extends the CORE_SIMD path and reuses the int4 head kernel"
#endif

// Interactive serial mode (host side: scripts/insiderllm/chat.py on the tools
// branch). 0 keeps the fixed benchmark run in setup(). 1 skips it and serves a
// line protocol on the native USB serial:
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
#define PREFILL_SKIP 1
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

#if CORE_SIMD
// ---- SIMD core matvecs --------------------------------------------------------
// Same arithmetic as matvec_par / matvec_q8 for one-group tensors: the int32
// row dots come from PIE kernels (exact), the float tail is the same
// expression, the >=128-row dual-core split is the same.
extern "C" void llm_dot128_s8(const int8_t *w, const int8_t *x, int32_t *out,
                              uint32_t rows);
extern "C" void llm_dot80_s8(const int8_t *w, const int8_t *x, int32_t *out,
                             uint32_t rows);
#define DOWN_PAD_COLS 80
static const int8_t *down_pad[LLM_MAX_LAYERS];   // 66-col down rows, zero-padded to 80

enum { K_DOT96, K_DOT128, K_DOT80, K_NONE };
static const char *const kernel_name[] = {"dot96", "dot128", "dot80", "none"};

static int core_kernel(const QT *t) {
  if (t->w8 == NULL || t->n_groups != 1) return K_NONE;
  if (t->cols == 96) return K_DOT96;
  if (t->cols == 128) return K_DOT128;
  if (t->cols == 66) {
    for (int l = 0; l < model.c.n_layers; l++)
      if (t == &model.down[l]) return down_pad[l] ? K_DOT80 : K_NONE;
  }
  return K_NONE;
}

// Exact int32 dots of rows [r0, r0+n) of t, n <= HEAD_BLOCK.
static void core_dots(const QT *t, int k, const int8_t *xq, int32_t *d, int r0, int n) {
  switch (k) {
    case K_DOT96:  llm_head_dot96_s8(t->w8 + (size_t)r0 * 96, xq, d, (uint32_t)n); break;
    case K_DOT128: llm_dot128_s8(t->w8 + (size_t)r0 * 128, xq, d, (uint32_t)n); break;
    default: {
      int l = 0;
      while (t != &model.down[l]) l++;
      llm_dot80_s8(down_pad[l] + (size_t)r0 * DOWN_PAD_COLS, xq, d, (uint32_t)n);
    }
  }
}

static void core_i8_range_simd(const QT *t, const int8_t *xq, float x_scale,
                               float *y, int row_begin, int row_end) {
  int k = core_kernel(t);
  int32_t d[HEAD_BLOCK];
  for (int r0 = row_begin; r0 < row_end; r0 += HEAD_BLOCK) {
    int n = row_end - r0 < HEAD_BLOCK ? row_end - r0 : HEAD_BLOCK;
    core_dots(t, k, xq, d, r0, n);
    for (int i = 0; i < n; i++) {
      float acc = 0.f;
      acc += (float)d[i] * t->scale8[r0 + i];
      y[r0 + i] = acc * x_scale;
    }
  }
}

#if CORE_INT4
// ---- packed int4 core ----------------------------------------------------------
// Each core tensor restaged at boot from the model's offset-8 codes into the
// int4 head layout: per row, ceil(cols/32) 16-byte chunks; chunk c byte k =
// column 32c+k (low nibble) and column 32c+16+k (high nibble), two's
// complement; columns past t->cols (down: 66..95) get zero weights. Values are
// unchanged, so the int32 dots, and with the same fp32 scale8, the outputs are
// identical. The int8 staging stays allocated for its scale8 (and is unread
// otherwise).
extern "C" void llm_dot128_s4(const uint8_t *w, const int8_t *x, int32_t *out,
                              uint32_t rows);
static const QT *core4_t[1 + 7 * LLM_MAX_LAYERS];
static const uint8_t *core4_w[1 + 7 * LLM_MAX_LAYERS];
static int core4_n;
static const uint8_t *core4_cur;   // weights of the matvec in flight (both cores)

static int core4_cols(const QT *t) { return t->cols > 96 ? 128 : 96; }

static const uint8_t *core4_find(const QT *t) {
  for (int i = 0; i < core4_n; i++) if (core4_t[i] == t) return core4_w[i];
  return NULL;
}

static void *ps_aligned16(size_t n);
static void core4_stage_one(const QT *t) {
  int pc = core4_cols(t), rb = pc / 2;
  size_t n = (size_t)t->rows * rb;
  uint8_t *w = (uint8_t *)ps_aligned16(n);
  if (!w) {
    Serial.printf("FATAL: required PSRAM allocation failed: int4 core (%u bytes)\n",
                  (unsigned)n);
    while (1) delay(1000);
  }
  for (int r = 0; r < t->rows; r++) {
    uint8_t *dst = w + (size_t)r * rb;
    for (int c = 0; c < pc / 32; c++)
      for (int k = 0; k < 16; k++) {
        int jl = 32 * c + k, jh = 32 * c + 16 + k;
        int lo = jl < t->cols ? head_code(t, r, jl) - 8 : 0;
        int hi = jh < t->cols ? head_code(t, r, jh) - 8 : 0;
        dst[16 * c + k] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
      }
  }
  core4_t[core4_n] = t;
  core4_w[core4_n++] = w;
}

static size_t core4_stage() {
  size_t bytes = 0;
  core4_n = 0;
  const QT *ts[1 + 7 * LLM_MAX_LAYERS];
  int nt = 0;
  ts[nt++] = &model.ple_model_proj;
  for (int l = 0; l < model.c.n_layers; l++) {
    ts[nt++] = &model.qkv[l];  ts[nt++] = &model.attn_proj[l];
    ts[nt++] = &model.gate[l]; ts[nt++] = &model.up[l];  ts[nt++] = &model.down[l];
    ts[nt++] = &model.ple_gate[l]; ts[nt++] = &model.ple_proj[l];
  }
  for (int i = 0; i < nt; i++) {
    core4_stage_one(ts[i]);
    bytes += (size_t)ts[i]->rows * (core4_cols(ts[i]) / 2);
  }
  return bytes;
}

// Exact int32 dots of rows [r0, r0+n) from packed int4 weights w4.
static void core4_dots(const QT *t, const uint8_t *w4, const int8_t *xq, int32_t *d,
                       int r0, int n) {
  if (core4_cols(t) == 128)
    llm_dot128_s4(w4 + (size_t)r0 * 64, xq, d, (uint32_t)n);
  else
    llm_head_dot96_s4(w4 + (size_t)r0 * 48, xq, d, (uint32_t)n);
}

static void core_i4_range_simd(const QT *t, const int8_t *xq, float x_scale,
                               float *y, int row_begin, int row_end) {
  const uint8_t *w4 = core4_cur;
  int32_t d[HEAD_BLOCK];
  for (int r0 = row_begin; r0 < row_end; r0 += HEAD_BLOCK) {
    int n = row_end - r0 < HEAD_BLOCK ? row_end - r0 : HEAD_BLOCK;
    core4_dots(t, w4, xq, d, r0, n);
    for (int i = 0; i < n; i++) {
      float acc = 0.f;
      acc += (float)d[i] * t->scale8[r0 + i];
      y[r0 + i] = acc * x_scale;
    }
  }
}

// Every row of every core tensor through its int4 kernel against the scalar
// int4 arithmetic of matvec_q8_range ((code - 8) * x straight from the model's
// codes), on a fixed activation with both int8 extremes and non-zero bytes past
// column 66 (down's padding must contribute nothing).
static bool core4_check() {
  static int8_t xt[128] __attribute__((aligned(16)));
  for (int j = 0; j < 128; j++)
    xt[j] = (j % 3 == 0) ? 127 : (j % 3 == 1) ? -127 : (int8_t)(j * 37 - 90);
  static const char *const name[3] = {"dot96_s4", "dot128_s4", "down96_s4"};
  int rows[3] = {0, 0, 0}, bad[3] = {0, 0, 0};
  for (int i = 0; i < core4_n; i++) {
    const QT *t = core4_t[i];
    const uint8_t *w4 = core4_w[i];
    int k = t->cols == 128 ? 1 : (t->cols == 96 ? 0 : 2);
    if ((t->cols != 96 && t->cols != 128 && t->cols != 66) || ((uintptr_t)w4 & 15) != 0 ||
        t->scale8 == NULL || t->n_groups != 1) {
      Serial.printf("FATAL: int4 core tensor %d unsupported (%dx%d, groups=%d, w4=%p)\n",
                    i, t->rows, t->cols, t->n_groups, w4);
      return false;
    }
    int32_t d[HEAD_BLOCK];
    for (int r0 = 0; r0 < t->rows; r0 += HEAD_BLOCK) {
      int n = t->rows - r0 < HEAD_BLOCK ? t->rows - r0 : HEAD_BLOCK;
      core4_dots(t, w4, xt, d, r0, n);
      for (int r = 0; r < n; r++) {
        int32_t ref = 0;
        for (int j = 0; j < t->cols; j++) ref += (head_code(t, r0 + r, j) - 8) * (int)xt[j];
        if (d[r] != ref) {
          if (bad[k] < 4) Serial.printf("int4 core mismatch: %s tensor %d row %d: %ld vs %ld\n",
                                        name[k], i, r0 + r, (long)d[r], (long)ref);
          bad[k]++;
        }
      }
      rows[k] += n;
    }
  }
  bool ok = true;
  for (int k = 0; k < 3; k++) {
    Serial.printf("simd core int4 selftest: %s %d/%d rows %s\n", name[k],
                  rows[k] - bad[k], rows[k], bad[k] ? "FAIL" : "ok");
    if (bad[k]) ok = false;
  }
  return ok;
}
#endif  // CORE_INT4

static void matvec_core_simd(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT] __attribute__((aligned(16)));
  float xs;
  if (core_kernel(t) == K_NONE) { matvec_par(t, x, y); return; }   // not expected
#if CORE_INT4
  range_fn fn = core_i4_range_simd;
  core4_cur = core4_find(t);
#else
  range_fn fn = core_i8_range_simd;
#endif
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  if (t->rows < 128) { fn(t, xq, xs, y, 0, t->rows); return; }
  job_fn = fn;
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  fn(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps_aligned16(size_t n) {
  void *p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}

// Zero-padded 80-byte-stride copies of each layer's 66-column down matrix.
static void stage_down_pad() {
  for (int l = 0; l < model.c.n_layers; l++) {
    const QT *t = &model.down[l];
    size_t n = (size_t)t->rows * DOWN_PAD_COLS;
    int8_t *p = (int8_t *)ps_aligned16(n);
    if (!p) {
      Serial.printf("FATAL: required PSRAM allocation failed: down pad (%u bytes)\n",
                    (unsigned)n);
      while (1) delay(1000);
    }
    memset(p, 0, n);
    for (int r = 0; r < t->rows; r++)
      memcpy(p + (size_t)r * DOWN_PAD_COLS, t->w8 + (size_t)r * t->cols, t->cols);
    down_pad[l] = p;
  }
}

// Every row of every core tensor through its kernel against llm_dot_i8, on a
// fixed activation with both int8 extremes and non-zero bytes past column 66.
static bool core_simd_check() {
  static int8_t xt[128] __attribute__((aligned(16)));
  for (int j = 0; j < 128; j++)
    xt[j] = (j % 3 == 0) ? 127 : (j % 3 == 1) ? -127 : (int8_t)(j * 37 - 90);
  const QT *ts[1 + 7 * LLM_MAX_LAYERS];
  int nt = 0;
  ts[nt++] = &model.ple_model_proj;
  for (int l = 0; l < model.c.n_layers; l++) {
    ts[nt++] = &model.qkv[l];  ts[nt++] = &model.attn_proj[l];
    ts[nt++] = &model.gate[l]; ts[nt++] = &model.up[l];  ts[nt++] = &model.down[l];
    ts[nt++] = &model.ple_gate[l]; ts[nt++] = &model.ple_proj[l];
  }
  int rows[3] = {0, 0, 0}, bad[3] = {0, 0, 0};
  for (int i = 0; i < nt; i++) {
    const QT *t = ts[i];
    int k = core_kernel(t);
    if (k == K_NONE) {
      Serial.printf("FATAL: no SIMD kernel for core tensor %d (%dx%d, groups=%d, w8=%p)\n",
                    i, t->rows, t->cols, t->n_groups, t->w8);
      return false;
    }
    const int8_t *base = (k == K_DOT80) ? NULL : t->w8;
    if (base && ((uintptr_t)base & 15) != 0) {
      Serial.printf("FATAL: core tensor %d not 16-byte aligned (%p)\n", i, base);
      return false;
    }
    int32_t d[HEAD_BLOCK];
    for (int r0 = 0; r0 < t->rows; r0 += HEAD_BLOCK) {
      int n = t->rows - r0 < HEAD_BLOCK ? t->rows - r0 : HEAD_BLOCK;
      core_dots(t, k, xt, d, r0, n);
      for (int r = 0; r < n; r++) {
        int32_t ref = llm_dot_i8(t->w8 + (size_t)(r0 + r) * t->cols, xt, t->cols);
        if (d[r] != ref) {
          if (bad[k] < 4) Serial.printf("simd core mismatch: %s tensor %d row %d: %ld vs %ld\n",
                                        kernel_name[k], i, r0 + r, (long)d[r], (long)ref);
          bad[k]++;
        }
      }
      rows[k] += n;
    }
  }
  bool ok = true;
  for (int k = 0; k < 3; k++) {
    Serial.printf("simd core selftest: %s %d/%d rows %s\n", kernel_name[k],
                  rows[k] - bad[k], rows[k], bad[k] ? "FAIL" : "ok");
    if (bad[k]) ok = false;
  }
  return ok;
}
#endif  // CORE_SIMD
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
  // logits: out_vocab floats, 99 KiB here, written by the head and read once by
  // the argmax per token. Upstream left them in PSRAM rather than spend a fifth
  // of internal SRAM; LOGITS_SRAM=1 spends it (about 309 -> 207 KB free on an
  // N16R8) and takes the head's 99 KiB of writes off PSRAM.
  {
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    size_t f0 = heap_caps_get_free_size(caps), b0 = heap_caps_get_largest_free_block(caps);
#if LOGITS_SRAM
    s.logits = (float *)sram_or_die((size_t)model.out_vocab * 4, "logits");
#else
    s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
#endif
    Serial.printf("logits -> %s %u B; internal free %u -> %u B, largest block %u -> %u B\n",
                  LOGITS_SRAM ? "SRAM  " : "PSRAM ", (unsigned)model.out_vocab * 4,
                  (unsigned)f0, (unsigned)heap_caps_get_free_size(caps),
                  (unsigned)b0, (unsigned)heap_caps_get_largest_free_block(caps));
  }
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

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
  Serial.printf("R ready vocab=%d seq_len=%d prefill_skip=%d\n",
                model.out_vocab, model.c.seq_len, PREFILL_SKIP);
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

  int pos = 0, tok = 0, decoded = 0;
  int64_t decode_us = 0, ttft_us = -1;
  prefill(ia_prompt, ia_n_prompt, &pos);
  llm_profile_reset(&s);
  for (int step = 0; step < n && pos < model.c.seq_len; step++) {
    if (temp == 0.f) {   // the benchmark's argmax, unchanged
      int best = 0; float bv = -1e30f;
      for (int v = 0; v < model.out_vocab; v++)
        if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
      tok = best;
    } else {
      tok = ia_sample(s.logits, model.out_vocab, temp, top_p);
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
  Serial.printf("D ttft_us=%lld decode_ms_per_tok=%.3f n=%d\n", (long long)ttft_us,
                decoded ? decode_us / 1000.0 / decoded : 0.0, decoded);
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

void setup() {
#if INTERACTIVE
  Serial.setRxBufferSize(IA_LINE_MAX);   // a whole P line fits while a run is busy
#endif
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
#if CORE_SIMD
  // 16-byte aligned staging: ee.vld.128 reads each row in place.
  int staged = llm_stage_core_int8_alloc(&model, ps_aligned16);
#else
  int staged = llm_stage_core_int8_alloc(&model, ps);
#endif
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
#if CORE_SIMD
  stage_down_pad();
#endif
#if CORE_INT4
  {
    size_t b4 = core4_stage();
    Serial.printf("core   -> PSRAM  packed int4, %d tensors, %u B\n", core4_n, (unsigned)b4);
  }
#endif
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
#if CORE_SIMD
#if CORE_INT4
    if (!core4_check()) {
      Serial.println("FATAL: int4 core selftest failed");
      while (1) delay(1000);
    }
    model.layer_matvec = matvec_core_simd;
    Serial.println("core   -> PIE int4 (head_dot96_s4, dot128_s4; down padded to 96)");
#else
    if (!core_simd_check()) {
      Serial.println("FATAL: simd core selftest failed");
      while (1) delay(1000);
    }
    model.layer_matvec = matvec_core_simd;
    Serial.println("core   -> PIE int8 (dot96, dot128, dot80)");
#endif
#else
    model.layer_matvec = matvec_par;
#endif
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

#if INTERACTIVE
  ia_reset_kv();
  ia_ready = true;
  ia_ready_line();
  return;
#endif

  // ---- generate ----
  Serial.print(">>> ");
  int n_prompt = sizeof(PROMPT_IDS) / sizeof(int);
  int pos = 0, tok = 0;
  int64_t decode_us = 0;
  int decoded = 0;

  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = PROMPT_IDS[i];
    emit(tok);
    forward_prompt(tok, pos++, i == n_prompt - 1);
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

void loop() {
#if INTERACTIVE
  ia_poll();
  delay(1);
#else
  delay(10000);
#endif
}
