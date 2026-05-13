/**
 * @file mnv_prng.h
 * @brief Xorshift32 PRNG for activation LUT blinding (v1.1)
 *
 * Used exclusively for masking LUT table indices during inference.
 * NOT a cryptographic PRNG — do not use for key generation.
 *
 * Properties:
 *   - Period: 2^32 - 1
 *   - ~4 AVR instructions per call (3 XOR-shifts)
 *   - Zero flash tables
 *   - State: one uint32_t in mnv_ctx_t
 */

#ifndef MNV_PRNG_H
#define MNV_PRNG_H

#include "mnv_types.h"

/**
 * @brief Advance PRNG state and return next 32-bit value.
 * Inline for performance — called once per neuron per layer.
 */
static inline uint32_t mnv_prng_next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * @brief Return a random uint8_t mask.
 */
static inline uint8_t mnv_prng_mask8(uint32_t *state)
{
    return (uint8_t)(mnv_prng_next(state) & 0xFFU);
}

#endif /* MNV_PRNG_H */
