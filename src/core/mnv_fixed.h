/**
 * @file mnv_fixed.h
 * @brief Q8 fixed-point arithmetic — internal header
 */

#ifndef MNV_FIXED_H
#define MNV_FIXED_H

#include "mnv_types.h"

mnv_act_t mnv_q8_clamp(mnv_acc_t x);
mnv_act_t mnv_q8_mul(mnv_act_t a, mnv_act_t b);
mnv_acc_t mnv_q8_dot(const mnv_weight_t *weights,
                     const mnv_act_t    *inputs,
                     uint16_t            len);
mnv_act_t mnv_q8_add_bias_clamp(mnv_acc_t acc, mnv_bias_t bias);
mnv_act_t mnv_apply_activation(mnv_act_fn_t fn, mnv_act_t x);

/* Individual activations exposed for testing */
mnv_act_t mnv_act_relu(mnv_act_t x);
mnv_act_t mnv_act_sigmoid(mnv_act_t x);
mnv_act_t mnv_act_tanh(mnv_act_t x);
mnv_act_t mnv_act_sign(mnv_act_t x);

#endif /* MNV_FIXED_H */
