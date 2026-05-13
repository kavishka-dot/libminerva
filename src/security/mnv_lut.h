/**
 * @file mnv_lut.h
 * @brief Blinded LUT activation functions — v1.1
 */

#ifndef MNV_LUT_H
#define MNV_LUT_H

#include "mnv_types.h"

int8_t mnv_lut_sigmoid_blinded(int8_t x, uint32_t *prng_state);
int8_t mnv_lut_tanh_blinded(int8_t x, uint32_t *prng_state);
int8_t mnv_lut_relu_blinded(int8_t x, uint32_t *prng_state);
int8_t mnv_lut_apply_blinded(mnv_act_fn_t fn, int8_t x, uint32_t *prng_state);

#endif /* MNV_LUT_H */
