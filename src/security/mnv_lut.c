/**
 * @file mnv_lut.c
 * @brief Input-blinded activation LUT access — v1.1 Law II hardening
 *
 * PROBLEM (v1.0):
 *   Naive LUT: out = TABLE[index]. Memory address leaks index through power.
 *
 * SOLUTION: Offset-masked linear scan
 *
 *   1. Generate random 8-bit mask M.
 *   2. Compute shifted target: target = (real_idx - M) & 0xFF
 *      This is the scan position i where the real entry will be selected.
 *   3. Scan the ENTIRE 256-entry table sequentially (i = 0..255):
 *      a. Read TABLE[(i + M) & 0xFF]  — address is i+M, a sliding window
 *      b. Select using CT: hit = 0xFF when i == target, else 0x00
 *      c. Accumulate: result = (result & ~hit) | (entry & hit)
 *
 *   The access sequence is i+M, i+M+1, ..., wrapping around.
 *   Every run accesses all 256 entries in a different order (random M).
 *   The entry at real_idx is accessed at scan position target = real_idx - M.
 *   Since M is random, target is uniformly distributed — leaks nothing.
 *
 *   Power trace shows 256 sequential-looking accesses starting at a random
 *   offset. The real index is unlinkable across calls.
 *
 * Cost: 256 reads per activation. ~512 cycles on ATmega @ 16 MHz.
 */

#include "mnv_lut.h"
#include "mnv_prng.h"

#define LUT_READ(ptr, idx) ((int8_t)pgm_read_byte(&(ptr)[(uint8_t)(idx)]))

/* ── Sigmoid LUT: sigmoid(x/127.0)*127, x=-128..127, idx=x+128 ─────────── */

static const int8_t MNV_SIGMOID_LUT_B[256] PROGMEM = {
    -63,-63,-63,-63,-63,-63,-63,-63,-63,-63,-63,-63,-63,-63,-63,-62,
    -62,-62,-62,-62,-62,-62,-61,-61,-61,-61,-61,-60,-60,-60,-60,-59,
    -59,-59,-58,-58,-58,-57,-57,-57,-56,-56,-55,-55,-55,-54,-54,-53,
    -53,-52,-52,-51,-51,-50,-50,-49,-49,-48,-47,-47,-46,-46,-45,-45,
    -44,-43,-43,-42,-41,-41,-40,-39,-39,-38,-37,-37,-36,-35,-35,-34,
    -33,-32,-32,-31,-30,-29,-29,-28,-27,-26,-25,-25,-24,-23,-22,-21,
    -21,-20,-19,-18,-17,-16,-16,-15,-14,-13,-12,-11,-10,-10, -9, -8,
     -7, -6, -5, -4, -3, -3, -2, -1,  0,  1,  2,  3,  3,  4,  5,  6,
      7,  8,  9, 10, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19, 20,
     21, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32,
     33, 34, 35, 35, 36, 37, 37, 38, 39, 39, 40, 41, 41, 42, 43, 43,
     44, 45, 45, 46, 46, 47, 47, 48, 49, 49, 50, 50, 51, 51, 52, 52,
     53, 53, 54, 54, 55, 55, 55, 56, 56, 57, 57, 57, 58, 58, 58, 59,
     59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62,
     62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
     63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63
};

/* ── Tanh LUT: tanh(x/64.0)*127, x=-128..127, idx=x+128 ─────────────────── */

static const int8_t MNV_TANH_LUT_B[256] PROGMEM = {
    -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-126,-126,-126,-126,
    -126,-125,-125,-125,-124,-124,-124,-123,-123,-122,-122,-121,-121,-120,-119,-119,
    -118,-117,-117,-116,-115,-114,-113,-113,-112,-111,-110,-109,-108,-107,-106,-104,
    -103,-102,-101,-100, -98, -97, -96, -94, -93, -91, -90, -88, -87, -85, -84, -82,
     -80, -79, -77, -75, -73, -72, -70, -68, -66, -64, -62, -60, -58, -56, -54, -52,
     -50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -27, -25, -23, -21, -19,
     -17, -15, -13, -10,  -8,  -6,  -4,  -2,   0,   2,   4,   6,   8,  10,  13,  15,
      17,  19,  21,  23,  25,  27,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,
      50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  73,  75,  77,  79,
      80,  82,  84,  85,  87,  88,  90,  91,  93,  94,  96,  97,  98, 100, 101, 102,
     103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 113, 114, 115, 116, 117, 117,
     118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 124, 124, 124, 125, 125, 125,
     126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127
};

/* ── Offset-masked linear scan ───────────────────────────────────────────── */

static int8_t blinded_lut_read(const int8_t *table,
                                uint8_t       real_idx,
                                uint32_t     *prng_state)
{
    uint8_t mask   = mnv_prng_mask8(prng_state);
    /* target: the scan position i where entry at real_idx will be selected */
    uint8_t target = (uint8_t)((uint16_t)real_idx - mask);
    int8_t  result = 0;

    for (uint16_t i = 0; i < 256U; i++) {
        /* Read at offset address: sliding window of width 256 */
        int8_t entry = LUT_READ(table, (uint8_t)((uint8_t)i + mask));

        /* CT select: hit=0xFF when i==target (i.e. i+mask == real_idx) */
        uint8_t eq  = (uint8_t)((uint8_t)i ^ target);
        uint8_t hit = (uint8_t)(((uint16_t)eq - 1U) >> 8U);

        result = (int8_t)((result & (int8_t)(~hit)) | (entry & (int8_t)hit));
    }
    return result;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int8_t mnv_lut_sigmoid_blinded(int8_t x, uint32_t *prng_state)
{
    return blinded_lut_read(MNV_SIGMOID_LUT_B,
                            (uint8_t)((int16_t)x + 128),
                            prng_state);
}

int8_t mnv_lut_tanh_blinded(int8_t x, uint32_t *prng_state)
{
    return blinded_lut_read(MNV_TANH_LUT_B,
                            (uint8_t)((int16_t)x + 128),
                            prng_state);
}

int8_t mnv_lut_relu_blinded(int8_t x, uint32_t *prng_state)
{
    /* Power-equalizing dummy scan */
    uint8_t mask = mnv_prng_mask8(prng_state);
    volatile int8_t dummy = 0;
    for (uint16_t i = 0; i < 256U; i++)
        dummy = (int8_t)(dummy ^ LUT_READ(MNV_SIGMOID_LUT_B, (uint8_t)((uint8_t)i + mask)));
    (void)dummy;
    /* Branchless ReLU */
    int8_t m = (int8_t)((int8_t)x >> 7);
    return (int8_t)(x & ~m);
}

int8_t mnv_lut_apply_blinded(mnv_act_fn_t fn, int8_t x, uint32_t *prng_state)
{
#if defined(MNV_ENABLE_BLINDED_LUT)
    switch (fn) {
        case MNV_ACT_RELU:    return mnv_lut_relu_blinded(x, prng_state);
        case MNV_ACT_SIGMOID: return mnv_lut_sigmoid_blinded(x, prng_state);
        case MNV_ACT_TANH:    return mnv_lut_tanh_blinded(x, prng_state);
        default:              (void)prng_state; return x;
    }
#else
    (void)prng_state;
    extern mnv_act_t mnv_apply_activation(mnv_act_fn_t, mnv_act_t);
    return mnv_apply_activation(fn, x);
#endif
}
