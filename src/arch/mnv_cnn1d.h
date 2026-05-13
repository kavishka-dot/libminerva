#ifndef MNV_CNN1D_H
#define MNV_CNN1D_H
#include "mnv_types.h"
#include "../security/mnv_chacha20.h"
mnv_status_t mnv_cnn1d_forward(mnv_ctx_t *ctx, const mnv_model_t *model,
                                const mnv_act_t *input, mnv_act_t *output,
                                mnv_chacha20_ctx_t *chacha);
#endif
