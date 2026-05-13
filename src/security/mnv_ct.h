/**
 * @file mnv_ct.h
 * @brief Constant-time primitives — internal header
 */

#ifndef MNV_CT_H
#define MNV_CT_H

#include "mnv_types.h"

void         mnv_secure_zero(void *ptr, size_t len);
uint8_t      mnv_ct_compare(const uint8_t *a, const uint8_t *b, size_t len);
void         mnv_canary_plant(mnv_ctx_t *ctx);
mnv_status_t mnv_canary_check(const mnv_ctx_t *ctx);
mnv_status_t mnv_ct_validate_input(const mnv_act_t *input, uint16_t len);
uint8_t      mnv_ct_argmax(const mnv_act_t *vec, uint16_t len);
mnv_status_t mnv_ct_confidence_check(const mnv_act_t *output, uint16_t len);

#endif /* MNV_CT_H */
