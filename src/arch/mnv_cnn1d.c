/**
 * @file mnv_cnn1d.c
 * @brief 1D CNN forward pass — v1.3 (stress-tested, fully bug-fixed)
 *
 * Bug fixes vs original:
 *   Bug 6:  MNV_CNN_DENSE_SHIFT replaces hardcoded >>7 for dense layer
 *   Bug 7:  arch-guarded #include in mnv_engine.c (fixed there)
 *   Bug 10: ct_offset tracks ciphertext position — was always reading from
 *           offset 0 (MLP engine uses ct+ct_offset pattern correctly;
 *           CNN1D was missing it entirely)
 *
 * Weight blob layout (must match compiler output exactly):
 *   [kernel_f0[0..K-1] | ... | kernel_fN[0..K-1]]   N*K bytes
 *   [conv_bias[0..N-1]]                              N bytes
 *   [dense_W.T row-major]                            OUTPUT*FLAT bytes
 *   [dense_bias[0..OUTPUT-1]]                        OUTPUT bytes
 */

#include "mnv_cnn1d.h"
#include "mnv_fixed.h"
#include "mnv_ct.h"
#include "mnv_chacha20.h"
#include "mnv_lut.h"
#include <string.h>

#if defined(MNV_ARCH_CNN1D)

/* ── Derived dimensions ────────────────────────────────────────────────────── */
#define MNV_CNN_CONV_LEN   (MNV_INPUT_SIZE - MNV_CNN_KERNEL_SIZE + 1U)
#define MNV_CNN_POOL_LEN   (MNV_CNN_CONV_LEN / MNV_CNN_POOL_SIZE)
#define MNV_CNN_FLAT_SIZE  (MNV_CNN_NUM_FILTERS * MNV_CNN_POOL_LEN)

/* Static scratch for one filter's pre-pool conv output */
static mnv_act_t conv_scratch[MNV_CNN_CONV_LEN];

/* ── Branchless max (CT, from v1.2 argmax fix) ─────────────────────────────── */
static inline mnv_act_t ct_max8(mnv_act_t a, mnv_act_t b)
{
    int16_t  diff = (int16_t)(int8_t)b - (int16_t)(int8_t)a;
    uint8_t  b_gt = (uint8_t)(~((uint8_t)((uint16_t)((int16_t)(diff-1)) >> 8U)));
    return (mnv_act_t)(((uint8_t)(int8_t)a & ~b_gt) |
                       ((uint8_t)(int8_t)b &  b_gt));
}

/* ── Single-filter: conv + ReLU + maxpool ──────────────────────────────────── */
static void conv1d_filter_forward(const mnv_weight_t *kernel,
                                  mnv_bias_t          bias,
                                  const mnv_act_t    *input,
                                  mnv_act_t          *pool_out,
                                  uint32_t           *prng_state)
{
    /* Convolution + ReLU */
    for (uint16_t i = 0U; i < MNV_CNN_CONV_LEN; i++) {
        mnv_acc_t acc = 0;
        for (uint16_t k = 0U; k < MNV_CNN_KERNEL_SIZE; k++) {
            acc += (mnv_acc_t)((int32_t)(int8_t)kernel[k] *
                               (int32_t)(int8_t)input[i + k]);
        }
        mnv_act_t pre = mnv_q8_add_bias_clamp(acc, bias);
#if defined(MNV_ENABLE_BLINDED_LUT)
        conv_scratch[i] = mnv_lut_apply_blinded(MNV_ACT_RELU, pre, prng_state);
#else
        conv_scratch[i] = mnv_apply_activation(MNV_ACT_RELU, pre);
        (void)prng_state;
#endif
    }

    /* MaxPool — branchless CT max */
    for (uint16_t p = 0U; p < MNV_CNN_POOL_LEN; p++) {
        mnv_act_t mv = conv_scratch[p * MNV_CNN_POOL_SIZE];
        for (uint16_t j = 1U; j < MNV_CNN_POOL_SIZE; j++)
            mv = ct_max8(mv, conv_scratch[p * MNV_CNN_POOL_SIZE + j]);
        pool_out[p] = mv;
    }
}

/* ── CNN1D forward pass ────────────────────────────────────────────────────── */
mnv_status_t mnv_cnn1d_forward(mnv_ctx_t          *ctx,
                                const mnv_model_t  *model,
                                const mnv_act_t    *input,
                                mnv_act_t          *output,
                                mnv_chacha20_ctx_t *chacha)
{
    mnv_status_t status;
    const uint8_t *ct     = model->encrypted_weights;
    uint16_t       ct_off = 0U;   /* FIX Bug 10: track ciphertext offset */

    mnv_act_t *feat_map = ctx->buf_a;   /* [MNV_CNN_FLAT_SIZE] */

    /* ── Conv block: all kernels first, then all biases ── */
    /* Blob layout: [K0 K1 ... KN] then [b0 b1 ... bN]        */
    /* Match compile_cnn1d.py which emits kernels flat, then biases flat */

    uint16_t kernel_bytes = (uint16_t)(MNV_CNN_KERNEL_SIZE * sizeof(mnv_weight_t));
    uint16_t all_kernels  = (uint16_t)(MNV_CNN_NUM_FILTERS * kernel_bytes);
    uint16_t all_biases   = (uint16_t)(MNV_CNN_NUM_FILTERS * sizeof(mnv_bias_t));

    /* Decrypt all kernels into a temporary staging area.
     * We use weight_scratch (large enough: MNV_OUTPUT_SIZE*FLAT_SIZE >= N*K) */
    mnv_chacha20_decrypt(chacha, ct + ct_off,
                         (uint8_t *)ctx->weight_scratch, all_kernels);
    ct_off += all_kernels;

    /* Decrypt all conv biases into a small local array */
    mnv_bias_t conv_bias[MNV_CNN_NUM_FILTERS];
    mnv_chacha20_decrypt(chacha, ct + ct_off,
                         (uint8_t *)conv_bias, all_biases);
    ct_off += all_biases;

    /* Apply each filter using its staged kernel */
    for (uint16_t f = 0U; f < MNV_CNN_NUM_FILTERS; f++) {
        conv1d_filter_forward(
            &ctx->weight_scratch[f * MNV_CNN_KERNEL_SIZE],
            conv_bias[f],
            input,
            feat_map + f * MNV_CNN_POOL_LEN,
            &ctx->prng_state);

        mnv_secure_zero(conv_scratch, sizeof(conv_scratch));

        status = mnv_canary_check(ctx);
        if (status != MNV_OK) goto fail;
    }

    mnv_secure_zero(conv_bias, sizeof(conv_bias));
    mnv_secure_zero(ctx->weight_scratch,
                    (uint16_t)(MNV_CNN_NUM_FILTERS * kernel_bytes));

    /* ── Dense output layer — one row at a time ── */
    uint16_t flat_bytes = (uint16_t)(MNV_CNN_FLAT_SIZE * sizeof(mnv_weight_t));

    for (uint16_t n = 0U; n < MNV_OUTPUT_SIZE; n++) {
        mnv_chacha20_decrypt(chacha, ct + ct_off,
                             (uint8_t *)ctx->weight_scratch, flat_bytes);
        ct_off += flat_bytes;

        mnv_acc_t acc = mnv_q8_dot(ctx->weight_scratch, feat_map,
                                    MNV_CNN_FLAT_SIZE);
        /* Dense shift: ceil(log2(FLAT_SIZE)) + 7 to stay in int8 range */
        output[n] = (mnv_act_t)mnv_q8_clamp(acc >> MNV_CNN_DENSE_SHIFT);

        mnv_secure_zero(ctx->weight_scratch, flat_bytes);
    }

    /* Decrypt and add dense biases */
    mnv_bias_t dense_bias[MNV_OUTPUT_SIZE];
    mnv_chacha20_decrypt(chacha, ct + ct_off,
                         (uint8_t *)dense_bias,
                         (uint16_t)(MNV_OUTPUT_SIZE * sizeof(mnv_bias_t)));
    ct_off += (uint16_t)(MNV_OUTPUT_SIZE * sizeof(mnv_bias_t));
    (void)ct_off;   /* suppress unused warning */

    for (uint16_t n = 0U; n < MNV_OUTPUT_SIZE; n++)
        output[n] = mnv_q8_clamp((mnv_acc_t)output[n] +
                                  (mnv_acc_t)dense_bias[n]);

    mnv_secure_zero(dense_bias, sizeof(dense_bias));

    status = mnv_canary_check(ctx);
    if (status != MNV_OK) goto fail;

    return MNV_OK;

fail:
    mnv_secure_zero(output,             MNV_OUTPUT_SIZE * sizeof(mnv_act_t));
    mnv_secure_zero(feat_map,           MNV_CNN_FLAT_SIZE);
    mnv_secure_zero(ctx->weight_scratch, sizeof(ctx->weight_scratch));
    mnv_secure_zero(conv_scratch,        sizeof(conv_scratch));
    return MNV_ERR_GLITCH;
}

#endif /* MNV_ARCH_CNN1D */
