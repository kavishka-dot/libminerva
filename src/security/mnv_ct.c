/**
 * @file mnv_ct.c
 * @brief Constant-time primitives — v1.2
 *
 * v1.2 fix: mnv_ct_argmax branchless select was broken — rewrote from scratch.
 * v1.2 fix: mnv_ct_confidence_check short-circuits to MNV_OK when
 *           MNV_MIN_CONFIDENCE == 0 (the new default).
 */

#include "mnv_ct.h"
#include <string.h>

void mnv_secure_zero(void *ptr, size_t len)
{
    volatile uint8_t *p = (volatile uint8_t *)ptr;
    while (len--) *p++ = 0;
}

uint8_t mnv_ct_compare(const uint8_t *a, const uint8_t *b, size_t len)
{
    uint8_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return diff;
}

void mnv_canary_plant(mnv_ctx_t *ctx)
{
    for (uint8_t i = 0; i < MNV_CANARY_COUNT; i++) {
        ctx->canary_pre[i]  = MNV_CANARY_VALUE;
        ctx->canary_post[i] = MNV_CANARY_VALUE;
    }
}

mnv_status_t mnv_canary_check(const mnv_ctx_t *ctx)
{
    uint32_t diff = 0;
    for (uint8_t i = 0; i < MNV_CANARY_COUNT; i++) {
        diff |= ctx->canary_pre[i]  ^ (uint32_t)MNV_CANARY_VALUE;
        diff |= ctx->canary_post[i] ^ (uint32_t)MNV_CANARY_VALUE;
    }
    return (diff == 0) ? MNV_OK : MNV_ERR_GLITCH;
}

mnv_status_t mnv_ct_validate_input(const mnv_act_t *input, uint16_t len)
{
    uint8_t bad = 0;
    for (uint16_t i = 0; i < len; i++) {
        int16_t v = (int16_t)(int8_t)input[i];
        bad |= (uint8_t)(((uint16_t)(v - (int16_t)MNV_Q_MIN)) >> 8U);
        bad |= (uint8_t)(((uint16_t)((int16_t)MNV_Q_MAX - v)) >> 8U);
    }
    return (bad == 0) ? MNV_OK : MNV_ERR_INPUT;
}

/**
 * Constant-time argmax for int8 vector.
 *
 * diff16 = vec[i] - max_val  (int16 to avoid int8 overflow at ±127/±128)
 * is_gt  = 0xFF if diff16 > 0, else 0x00
 *        = ~( (uint8_t)( (uint16_t)(diff16 - 1) >> 8 ) )
 *   When diff16 > 0: diff16-1 >= 0, top byte of uint16 = 0x00, ~0x00 = 0xFF
 *   When diff16 <= 0: diff16-1 < 0, uint16 top byte = 0xFF, ~0xFF = 0x00
 */
uint8_t mnv_ct_argmax(const mnv_act_t *vec, uint16_t len)
{
    int8_t  max_val = (int8_t)vec[0];
    uint8_t max_idx = 0U;

    for (uint16_t i = 1U; i < len; i++) {
        int16_t diff16 = (int16_t)(int8_t)vec[i] - (int16_t)max_val;
        uint8_t is_gt  = (uint8_t)(~((uint8_t)((uint16_t)((int16_t)(diff16 - 1)) >> 8U)));

        max_val = (int8_t) (((uint8_t)max_val         & ~is_gt) |
                             ((uint8_t)(int8_t)vec[i]  &  is_gt));
        max_idx = (uint8_t)((max_idx                  & ~is_gt) |
                             ((uint8_t)i               &  is_gt));
    }
    return max_idx;
}

mnv_status_t mnv_ct_confidence_check(const mnv_act_t *output, uint16_t len)
{
#if MNV_MIN_CONFIDENCE == 0
    (void)output; (void)len;
    return MNV_OK;
#else
    int8_t max_val = (int8_t)output[0];
    for (uint16_t i = 1U; i < len; i++) {
        int16_t d  = (int16_t)(int8_t)output[i] - (int16_t)max_val;
        uint8_t gt = (uint8_t)(~((uint8_t)((uint16_t)((int16_t)(d - 1)) >> 8U)));
        max_val = (int8_t)(((uint8_t)max_val          & ~gt) |
                            ((uint8_t)(int8_t)output[i] &  gt));
    }
    return ((uint8_t)max_val >= (uint8_t)MNV_MIN_CONFIDENCE) ? MNV_OK
                                                              : MNV_ERR_CONFIDENCE;
#endif
}
