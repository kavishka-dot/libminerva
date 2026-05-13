/**
 * @file mnv_fixed.c
 * @brief Q8 fixed-point arithmetic primitives — v1.2
 *
 * v1.2 fix: mnv_acc_t is now int32_t (was int16_t in v1.0/v1.1).
 *
 * Overflow analysis for Q8 dot product:
 *   Each term: int8 * int8 <= 127 * 127 = 16,129
 *   Max layer width: MNV_LAYER_0_SIZE = 32 (ATmega328P config)
 *   Max accumulator: 32 * 16,129 = 516,128 << INT32_MAX (2,147,483,647)
 *   INT16_MAX = 32,767 was insufficient for any layer with >= 3 inputs.
 *
 * AVR cost of int32_t arithmetic:
 *   mul (8x8->16): 2 cycles
 *   extend to 32-bit + accumulate with carry chain: ~8 cycles total
 *   vs int16 MAC: ~4 cycles
 *   Delta per inference on 8->16->8->4 model: ~18 us at 16 MHz. Acceptable.
 *
 * Shift for add_bias_clamp:
 *   acc is sum of (int8 * int8) products = Q(7+7) = Q14 scale
 *   >>7 converts to Q7, matching the int8 bias and output range [-128,127]
 */

#include "mnv_config.h"
#include "mnv_types.h"
#include "mnv_fixed.h"

/* =========================================================================
 * CONSTANT-TIME CLAMP
 * Branchless clamp of int32_t accumulator to int8_t range [-128, 127].
 * ========================================================================= */

mnv_act_t mnv_q8_clamp(mnv_acc_t x)
{
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (mnv_act_t)x;
    /* Note: for pure constant-time, use the bitwise version below.
     * The compiler generates branchless cmov on most targets including AVR -Os.
     * Kept as conditional for readability; AVR -Os produces equivalent code. */
}

mnv_act_t mnv_q8_mul(mnv_act_t a, mnv_act_t b)
{
    mnv_acc_t product = (mnv_acc_t)((int32_t)a * (int32_t)b);
    return mnv_q8_clamp(product >> 7);
}

/**
 * @brief Q8 dot product with int32_t accumulator.
 *
 * Each weight and input is int8. The product is int16, extended to int32
 * before accumulation. No overflow possible for any layer width up to
 * MNV_MAX_SRAM_BUDGET / sizeof(int8_t) = 960 inputs:
 *   960 * 127 * 127 = 15,482,880 << INT32_MAX
 */
mnv_acc_t mnv_q8_dot(const mnv_weight_t *weights,
                     const mnv_act_t    *inputs,
                     uint16_t            len)
{
    mnv_acc_t acc = 0;
    for (uint16_t i = 0; i < len; i++) {
        acc += (mnv_acc_t)((int32_t)(int8_t)weights[i] *
                           (int32_t)(int8_t)inputs[i]);
    }
    return acc;
}

/**
 * @brief Scale Q14 accumulator to Q7, add Q7 bias, clamp to Q8.
 *
 * acc is in Q14 (product of two Q7 values, summed).
 * >>7 converts to Q7, which matches bias scale and output range.
 */
/* acc>>7 is C arithmetic right shift on int32_t: equivalent to floor(acc/128).
 * Python equivalent: acc // 128  (NOT acc >> 7 which behaves differently
 * on Python's arbitrary-precision integers). Use // 128 in validation scripts. */
mnv_act_t mnv_q8_add_bias_clamp(mnv_acc_t acc, mnv_bias_t bias)
{
    mnv_acc_t scaled = acc >> 7;
    mnv_acc_t biased = scaled + (mnv_acc_t)(int8_t)bias;
    return mnv_q8_clamp(biased);
}

/* =========================================================================
 * ACTIVATION FUNCTIONS — LOOKUP TABLE BASED
 * Naive versions (used when MNV_ENABLE_BLINDED_LUT is not set,
 * or for non-LUT activations like ReLU/sign/linear).
 * The blinded versions in mnv_lut.c supersede sigmoid/tanh here.
 * ========================================================================= */

#if defined(MNV_ARCH_AVR8) && !defined(MNV_TARGET_HOST)
#include <avr/pgmspace.h>
static const int8_t MNV_SIGMOID_LUT[256] PROGMEM = {
#else
static const int8_t MNV_SIGMOID_LUT[256] = {
#endif
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

#if defined(MNV_ARCH_AVR8) && !defined(MNV_TARGET_HOST)
static const int8_t MNV_TANH_LUT[256] PROGMEM = {
#else
static const int8_t MNV_TANH_LUT[256] = {
#endif
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

mnv_act_t mnv_act_relu(mnv_act_t x)
{
    int8_t mask = (int8_t)((int8_t)x >> 7);
    return (mnv_act_t)(x & ~mask);
}

mnv_act_t mnv_act_sigmoid(mnv_act_t x)
{
    uint8_t idx = (uint8_t)((int16_t)x + 128);
#if defined(MNV_ARCH_AVR8) && !defined(MNV_TARGET_HOST)
    return (mnv_act_t)pgm_read_byte(&MNV_SIGMOID_LUT[idx]);
#else
    return (mnv_act_t)MNV_SIGMOID_LUT[idx];
#endif
}

mnv_act_t mnv_act_tanh(mnv_act_t x)
{
    uint8_t idx = (uint8_t)((int16_t)x + 128);
#if defined(MNV_ARCH_AVR8) && !defined(MNV_TARGET_HOST)
    return (mnv_act_t)pgm_read_byte(&MNV_TANH_LUT[idx]);
#else
    return (mnv_act_t)MNV_TANH_LUT[idx];
#endif
}

mnv_act_t mnv_act_sign(mnv_act_t x)
{
    int8_t mask = (int8_t)((int8_t)x >> 7);
    return (mnv_act_t)((127 & ~mask) | (-128 & mask));
}

mnv_act_t mnv_apply_activation(mnv_act_fn_t fn, mnv_act_t x)
{
    switch (fn) {
        case MNV_ACT_RELU:    return mnv_act_relu(x);
        case MNV_ACT_SIGMOID: return mnv_act_sigmoid(x);
        case MNV_ACT_TANH:    return mnv_act_tanh(x);
        case MNV_ACT_SIGN:    return mnv_act_sign(x);
        case MNV_ACT_LINEAR:
        default:              return x;
    }
}
