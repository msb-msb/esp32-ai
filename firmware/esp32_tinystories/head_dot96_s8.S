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

/* Packed int4 variant of the same 96-column head.
 *
 *   void llm_head_dot96_s4(const uint8_t *w, const int8_t *x, int32_t *out,
 *                          uint32_t rows);
 *
 * Each row is 48 bytes = three 16-byte chunks. In chunk c, byte k holds
 * element 32c+k in its low nibble and element 32c+16+k in its high nibble,
 * both two's-complement int4 (restaged at boot from the file's offset-8
 * codes). With M = 0xF0 in every byte:
 *   (w & M)         read as int8 = 16 * hi
 *   ((w << 4) & M)  read as int8 = 16 * lo   (32-bit lane shift; the mask
 *                                             drops bits from the next byte)
 * so ACCX = 16 * dot exactly, and out[r] = ACCX >> 4 is exact. x stays in
 * q0..q5 exactly as for the int8 kernel: chunk c pairs lo with q(2c) and hi
 * with q(2c+1). q6 holds M, q7 is the work register; each chunk is loaded
 * twice (the second load hits the line the first one filled).
 * w and x must be 16-byte aligned. SAR is clobbered (scratch in the ABI).
 */

.section .iram1,"ax"
.align 4
.global llm_head_dot96_s4
.type llm_head_dot96_s4,@function
llm_head_dot96_s4:
    entry a1,16

    ee.vld.128.ip q0,a3,16
    ee.vld.128.ip q1,a3,16
    ee.vld.128.ip q2,a3,16
    ee.vld.128.ip q3,a3,16
    ee.vld.128.ip q4,a3,16
    ee.vld.128.ip q5,a3,16

    /* q6 = 0xF0 in all 16 bytes, built without a literal pool entry */
    movi a8,0xF0
    slli a9,a8,8
    or a8,a8,a9
    slli a9,a8,16
    or a8,a8,a9
    ee.movi.32.q q6,a8,0
    ee.movi.32.q q6,a8,1
    ee.movi.32.q q6,a8,2
    ee.movi.32.q q6,a8,3
    ssai 4                      /* ee.vsl.32 shifts left by SAR */

    loopnez a5,.Lhead4_rows_end
    ee.zero.accx
    /* chunk 0: elements 0..15 (lo) and 16..31 (hi) */
    ee.vld.128.ip q7,a2,0
    ee.andq q7,q7,q6
    ee.vmulas.s8.accx q1,q7
    ee.vld.128.ip q7,a2,16
    ee.vsl.32 q7,q7
    ee.andq q7,q7,q6
    ee.vmulas.s8.accx q0,q7
    /* chunk 1: 32..47 / 48..63 */
    ee.vld.128.ip q7,a2,0
    ee.andq q7,q7,q6
    ee.vmulas.s8.accx q3,q7
    ee.vld.128.ip q7,a2,16
    ee.vsl.32 q7,q7
    ee.andq q7,q7,q6
    ee.vmulas.s8.accx q2,q7
    /* chunk 2: 64..79 / 80..95 */
    ee.vld.128.ip q7,a2,0
    ee.andq q7,q7,q6
    ee.vmulas.s8.accx q5,q7
    ee.vld.128.ip q7,a2,16
    ee.vsl.32 q7,q7
    ee.andq q7,q7,q6
    ee.vmulas.s8.accx q4,q7
    rur.accx_0 a8
    srai a8,a8,4
    s32i a8,a4,0
    addi a4,a4,4
.Lhead4_rows_end:
    retw.n
.size llm_head_dot96_s4,.-llm_head_dot96_s4
