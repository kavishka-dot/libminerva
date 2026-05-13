/**
 * @file mnv_bnn.h
 */
#ifndef MNV_BNN_H
#define MNV_BNN_H

#include "mnv_types.h"
#include "../security/mnv_chacha20.h"

mnv_status_t mnv_bnn_forward(mnv_ctx_t *ctx, const mnv_model_t *model,
                              const mnv_act_t *input, mnv_act_t *output,
                              mnv_chacha20_ctx_t *chacha);

#endif /* MNV_BNN_H */
