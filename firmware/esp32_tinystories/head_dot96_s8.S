/* MIT License. Original esp32-ai (c) 2026 Viacheslav Sierbov. Changes (c) 2026 Mark Bartlett / InsiderLLM. */

/* ESP32-S3 int8 dot products for the 96-column output head.
 *
 * C interface (extern "C" from C++):
 *   void llm_head_dot96_s8(const int8_t *w, const int8_t *x, int32_t *out,
 *                          uint32_t rows);
 *
 * out[r] = sum_{j<96} w[r*96 + j] * x[j] for r in [0, rows), exactly: the
 * 40-bit ACCX holds every int8 product sum, and |sum| <= 96*128*128 fits the
 * low 32 bits read back with rur.accx_0.
 *
 * w and x must be 16-byte aligned (ee.vld.128 ignores the low four address
 * bits); every row is 96 = 6*16 bytes, so rows stay aligned. The caller checks
 * this. x is loaded once into q0..q5 and reused for every row; q6/q7 stream
 * the weights. q0..q7 and ACCX are scratch. Kept in IRAM.
 */

.section .iram1,"ax"
.align 4
.global llm_head_dot96_s8
.type llm_head_dot96_s8,@function
llm_head_dot96_s8:
    entry a1,16

    /* activation: 96 bytes -> q0..q5 */
    ee.vld.128.ip q0,a3,16
    ee.vld.128.ip q1,a3,16
    ee.vld.128.ip q2,a3,16
    ee.vld.128.ip q3,a3,16
    ee.vld.128.ip q4,a3,16
    ee.vld.128.ip q5,a3,16

    loopnez a5,.Lhead_rows_end
    ee.zero.accx
    ee.vld.128.ip q6,a2,16
    ee.vld.128.ip q7,a2,16
    ee.vmulas.s8.accx.ld.ip q6,a2,16,q0,q6
    ee.vmulas.s8.accx.ld.ip q7,a2,16,q1,q7
    ee.vmulas.s8.accx.ld.ip q6,a2,16,q2,q6
    ee.vmulas.s8.accx.ld.ip q7,a2,16,q3,q7
    ee.vmulas.s8.accx q4,q6
    ee.vmulas.s8.accx q5,q7
    rur.accx_0 a8
    s32i a8,a4,0
    addi a4,a4,4
.Lhead_rows_end:
    retw.n
.size llm_head_dot96_s8,.-llm_head_dot96_s8
