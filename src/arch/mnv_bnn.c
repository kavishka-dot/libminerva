/**
 * @file mnv_bnn.c
 * @brief Binary Neural Network (BNN) forward pass
 *
 * Weights ∈ {-1, +1}, packed 8 per byte (1 = +1, 0 = -1).
 * Activations ∈ {-1, +1} after sign activation.
 *
 * Multiply-accumulate becomes:
 *   XNOR(weight_bit, activation_bit) → popcount → scale
 *
 * This is:
 *   - ~58× fewer operations than Q8 on AVR
 *   - Inherently constant-time (popcount is data-independent in time)
 *   - Maximally compact: 8 weights per byte
 *
 * On ATtiny85 with 2KB weight budget: ~16K binary weights feasible.
 *
 * Weight packing: weights[byte] bit k = 1 → w = +1, 0 → w = -1
 * Packed column-major for XNOR efficiency.
 */

#include "mnv_bnn.h"
#include "../core/mnv_fixed.h"
#include "../security/mnv_ct.h"
#include "../security/mnv_chacha20.h"
#include <string.h>

#if defined(MNV_ARCH_BNN)

/* =========================================================================
 * POPCOUNT — constant-time on all targets
 * ========================================================================= */

static uint8_t mnv_popcount8(uint8_t x)
{
    /* Brian Kernighan's algorithm — but on AVR, use parallel bit sum */
    x = x - ((x >> 1) & 0x55u);
    x = (x & 0x33u) + ((x >> 2) & 0x33u);
    x = (x + (x >> 4)) & 0x0Fu;
    return x;
}

/* =========================================================================
 * BNN DOT PRODUCT
 * Input activations are packed 1 bit per value (1 = +1, 0 = -1).
 * ========================================================================= */

/**
 * @brief Binary dot product: sum_i XNOR(w_i, a_i) scaled to {-N..+N}
 *
 * @param weights_packed  Packed weight bytes [ceil(len/8)]
 * @param acts_packed     Packed activation bytes [ceil(len/8)]
 * @param len             Number of binary values
 * @return Signed accumulator in range [-len, +len]
 */
static int16_t bnn_dot_packed(const uint8_t *weights_packed,
                               const uint8_t *acts_packed,
                               uint16_t       len)
{
    uint16_t n_bytes = (len + 7u) / 8u;
    int16_t  acc     = 0;

    for (uint16_t i = 0; i < n_bytes; i++) {
        /* XNOR: agreement = 1, disagreement = 0 */
        uint8_t xnor = (uint8_t)~(weights_packed[i] ^ acts_packed[i]);
        uint8_t agrees = mnv_popcount8(xnor);        /* count +1 agreements */
        uint8_t n_in_byte = (i < n_bytes - 1) ? 8u : (uint8_t)(((len - 1u) % 8u) + 1u);
        uint8_t disagrees = n_in_byte - agrees;      /* count -1 agreements */
        acc += (int16_t)agrees - (int16_t)disagrees;
    }
    return acc;
}

/* =========================================================================
 * ACTIVATION PACKING
 * Pack int8 activations (sign bit) into bit array.
 * 1 = positive (≥0), 0 = negative (<0)
 * ========================================================================= */

static void pack_activations(const mnv_act_t *acts, uint8_t *packed, uint16_t len)
{
    uint16_t n_bytes = (len + 7u) / 8u;
    for (uint16_t b = 0; b < n_bytes; b++) {
        uint8_t byte = 0;
        for (uint8_t bit = 0; bit < 8u && (b * 8u + bit) < len; bit++) {
            /* 1 if non-negative */
            byte |= (uint8_t)(((uint8_t)(~((uint8_t)acts[b * 8u + bit] >> 7u))) & 1u) << bit;
        }
        packed[b] = byte;
    }
}

/* =========================================================================
 * BNN FORWARD PASS
 * ========================================================================= */

/* Packed activation scratch — ceil(MNV_LAYER_0_SIZE / 8) bytes */
#define MNV_BNN_PACKED_BYTES  ((MNV_LAYER_0_SIZE + 7u) / 8u)
static uint8_t packed_src[MNV_BNN_PACKED_BYTES];
static uint8_t packed_weights[MNV_BNN_PACKED_BYTES];

mnv_status_t mnv_bnn_forward(mnv_ctx_t          *ctx,
                              const mnv_model_t  *model,
                              const mnv_act_t    *input,
                              mnv_act_t          *output,
                              mnv_chacha20_ctx_t *chacha)
{
    mnv_status_t status;

    mnv_act_t *src = ctx->buf_a;
    mnv_act_t *dst = ctx->buf_b;

    /* Copy input */
    for (uint16_t i = 0; i < MNV_INPUT_SIZE; i++) src[i] = input[i];
    uint16_t src_size = MNV_INPUT_SIZE;

    for (uint8_t layer = 0; layer < model->num_layers; layer++) {
        const mnv_layer_desc_t *ld = &model->layers[layer];
        uint16_t in_sz    = ld->input_size;
        uint16_t out_sz   = ld->output_size;
        uint16_t w_bytes  = (in_sz * out_sz + 7u) / 8u;

        /* Decrypt packed weights */
        mnv_chacha20_decrypt(chacha, model->encrypted_weights,
                             (uint8_t *)ctx->weight_scratch, w_bytes);

        /* Pack input activations */
        pack_activations(src, packed_src, in_sz);

        /* For each output neuron */
        for (uint16_t n = 0; n < out_sz; n++) {
            uint16_t w_byte_offset = (n * in_sz) / 8u;
            /* Extract packed weights for this neuron */
            uint16_t neuron_bytes = (in_sz + 7u) / 8u;
            memcpy(packed_weights,
                   (uint8_t *)ctx->weight_scratch + w_byte_offset,
                   neuron_bytes);

            int16_t acc = bnn_dot_packed(packed_weights, packed_src, in_sz);

            /* Scale to Q8: acc ∈ [-in_sz, +in_sz], map to [-127, +127] */
            int16_t scaled = (int16_t)((int32_t)acc * 127 / (int16_t)in_sz);
            mnv_act_t pre_act = (mnv_act_t)(scaled > 127 ? 127 : (scaled < -128 ? -128 : scaled));

            /* BNN uses sign activation for hidden layers, linear for output */
            dst[n] = mnv_apply_activation(ld->activation, pre_act);
        }

        mnv_secure_zero(ctx->weight_scratch, w_bytes);
        mnv_secure_zero(packed_src,     sizeof(packed_src));
        mnv_secure_zero(packed_weights, sizeof(packed_weights));

        status = mnv_canary_check(ctx);
        if (status != MNV_OK) return MNV_ERR_GLITCH;

        mnv_act_t *tmp = src; src = dst; dst = tmp;
        src_size = out_sz;
        (void)src_size;
    }

    for (uint16_t i = 0; i < MNV_OUTPUT_SIZE; i++) output[i] = src[i];
    return MNV_OK;
}

#endif /* MNV_ARCH_BNN */
