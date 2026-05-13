/**
 * @file mnv_mlp.c
 * @brief MLP forward pass — v1.1
 *
 * Fixed from v1.0:
 *   - ChaCha20 offset arithmetic was stubbed with incorrect pointer math.
 *     The encrypted blob is a flat byte stream. The ChaCha20 context tracks
 *     the stream position internally via (counter, block_pos). We simply
 *     call mnv_chacha20_decrypt() sequentially — the context advances
 *     automatically. No manual pointer arithmetic needed.
 *
 *   - Weight and bias scratch buffers are now decrypted in sequence:
 *       weights_layer_0 | biases_layer_0 | weights_layer_1 | biases_layer_1 ...
 *     The compiler emits the ciphertext in this exact order.
 *
 * v1.1 addition:
 *   - Activation dispatch goes through mnv_lut_apply_blinded() when
 *     MNV_ENABLE_BLINDED_LUT is defined, passing ctx->prng_state.
 */

#include "mnv_mlp.h"
#include "mnv_fixed.h"
#include "mnv_ct.h"
#include "mnv_chacha20.h"
#include "mnv_lut.h"
#include <string.h>

#if defined(MNV_ARCH_MLP)

mnv_status_t mnv_mlp_forward(mnv_ctx_t          *ctx,
                              const mnv_model_t  *model,
                              const mnv_act_t    *input,
                              mnv_act_t          *output,
                              mnv_chacha20_ctx_t *chacha)
{
    mnv_status_t status;

    /* Ping-pong activation buffers */
    mnv_act_t *src = ctx->buf_a;
    mnv_act_t *dst = ctx->buf_b;

    /* Copy input into src */
    for (uint16_t i = 0; i < MNV_INPUT_SIZE; i++) src[i] = input[i];

    /* Encrypted blob stream position is managed entirely by the chacha context.
     * Each call to mnv_chacha20_decrypt() advances the keystream position by
     * exactly `len` bytes. The compiler emits weights then biases per layer,
     * so we just call decrypt in the same order. */
    const uint8_t *ct = model->encrypted_weights;  /* base pointer */
    uint16_t       ct_offset = 0;                  /* bytes consumed so far */

    for (uint8_t layer = 0; layer < model->num_layers; layer++) {

        const mnv_layer_desc_t *ld = &model->layers[layer];
        uint16_t in_sz        = ld->input_size;
        uint16_t out_sz       = ld->output_size;
        uint16_t weight_bytes = (uint16_t)((uint32_t)in_sz * out_sz * sizeof(mnv_weight_t));
        uint16_t bias_bytes   = (uint16_t)(out_sz * sizeof(mnv_bias_t));

        /* Decrypt weights: row-major, weight[out_neuron * in_sz + in_neuron] */
        mnv_chacha20_decrypt(chacha,
                             ct + ct_offset,
                             (uint8_t *)ctx->weight_scratch,
                             weight_bytes);
        ct_offset += weight_bytes;

        /* Decrypt biases */
        mnv_bias_t bias_scratch[MNV_LAYER_0_SIZE];
        mnv_chacha20_decrypt(chacha,
                             ct + ct_offset,
                             (uint8_t *)bias_scratch,
                             bias_bytes);
        ct_offset += bias_bytes;

        /* Compute layer output */
        for (uint16_t n = 0; n < out_sz; n++) {
            mnv_acc_t acc = mnv_q8_dot(
                &ctx->weight_scratch[n * in_sz],
                src,
                in_sz);

            mnv_act_t pre_act = mnv_q8_add_bias_clamp(acc, bias_scratch[n]);

            /* v1.1: blinded LUT activation (Law II) */
#if defined(MNV_ENABLE_BLINDED_LUT)
            dst[n] = mnv_lut_apply_blinded(ld->activation, pre_act, &ctx->prng_state);
#else
            dst[n] = mnv_apply_activation(ld->activation, pre_act);
#endif
        }

        /* Zero weight and bias scratch immediately after use */
        mnv_secure_zero(ctx->weight_scratch, weight_bytes);
        mnv_secure_zero(bias_scratch, bias_bytes);

        /* Canary check after every layer */
        status = mnv_canary_check(ctx);
        if (status != MNV_OK) {
            mnv_secure_zero(dst, out_sz);
            return MNV_ERR_GLITCH;
        }

        /* Swap ping-pong buffers */
        mnv_act_t *tmp = src; src = dst; dst = tmp;
    }

    /* Copy result to output */
    for (uint16_t i = 0; i < MNV_OUTPUT_SIZE; i++) output[i] = src[i];

    return MNV_OK;
}

#endif /* MNV_ARCH_MLP */
