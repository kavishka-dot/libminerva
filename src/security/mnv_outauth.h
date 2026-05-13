/**
 * @file mnv_outauth.h
 * @brief Output authentication — v1.1
 */

#ifndef MNV_OUTAUTH_H
#define MNV_OUTAUTH_H

#include "mnv_types.h"

/* Add inference_counter to ctx — declared here, added to mnv_ctx_t in types */
/* Note: inference_counter is in mnv_ctx_t as of v1.1 */

void         mnv_outauth_compute(mnv_ctx_t *ctx,
                                  const uint8_t *device_key,
                                  const mnv_act_t *output,
                                  const mnv_act_t *input);

mnv_status_t mnv_outauth_verify(const mnv_ctx_t *ctx,
                                 const uint8_t *device_key,
                                 const mnv_act_t *output,
                                 const mnv_act_t *input);

#endif /* MNV_OUTAUTH_H */
