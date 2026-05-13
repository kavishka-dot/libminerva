/**
 * @file mnv_engine.c
 * @brief Minerva inference engine — v1.1
 *
 * Changes from v1.0:
 *   - Fixed: ChaCha20 offset tracking was stubbed — now fully wired
 *   - Fixed: model pointer stubs removed, mnv_run() demoted to error shim
 *   - Added: mnv_seed_prng() for hardware entropy injection
 *   - Added: blinded LUT dispatch via prng_state in ctx
 *   - Added: output MAC computation (mnv_outauth) after every inference
 *   - Added: mnv_verify_output_with_key(), mnv_get_output_mac()
 */

#include "minerva.h"
#include "mnv_blake2s.h"
#include "mnv_chacha20.h"
#include "mnv_ct.h"
#include "mnv_outauth.h"
#include "mnv_prng.h"
#if defined(MNV_ARCH_MLP)
#include "mnv_mlp.h"
#elif defined(MNV_ARCH_CNN1D)
#include "mnv_cnn1d.h"
#elif defined(MNV_ARCH_BNN)
#include "mnv_bnn.h"
#endif
#include <string.h>

static void engine_chacha_init(mnv_chacha20_ctx_t *chacha,
                                const mnv_model_t  *model)
{
    mnv_chacha20_init(chacha, model->key, model->crypto->iv, 0U);
}

static mnv_status_t engine_forward(mnv_ctx_t          *ctx,
                                    const mnv_model_t  *model,
                                    const mnv_act_t    *input,
                                    mnv_act_t          *output,
                                    mnv_chacha20_ctx_t *chacha)
{
#if defined(MNV_ARCH_MLP)
    return mnv_mlp_forward(ctx, model, input, output, chacha);
#elif defined(MNV_ARCH_CNN1D)
    return mnv_cnn1d_forward(ctx, model, input, output, chacha);
#elif defined(MNV_ARCH_BNN)
    return mnv_bnn_forward(ctx, model, input, output, chacha);
#else
    (void)ctx; (void)model; (void)input; (void)output; (void)chacha;
    return MNV_ERR_CONFIG;
#endif
}

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

mnv_status_t mnv_init(mnv_ctx_t *ctx, const mnv_model_t *model)
{
    if (!ctx || !model) return MNV_ERR_NULL;
    mnv_secure_zero(ctx, sizeof(mnv_ctx_t));
    if (model->version != MNV_ABI_VERSION)   return MNV_ERR_CONFIG;
    /* Layer count check only for MLP (CNN1D uses num_layers=0) */
#if defined(MNV_ARCH_MLP)
    if (model->num_layers == 0 ||
        model->num_layers > MNV_NUM_LAYERS)  return MNV_ERR_CONFIG;
#endif

    mnv_canary_plant(ctx);
    ctx->prng_state        = (uint32_t)MNV_PRNG_SEED_DEFAULT;
    ctx->inference_counter = 0U;

    /* Law I — integrity before anything.
     * On AVR with PROGMEM weights, read ciphertext in 64B chunks
     * via pgm_read_byte to avoid copying 492B to SRAM. */
    mnv_status_t s;
    {
        mnv_blake2s_ctx_t bctx;
        mnv_blake2s_init(&bctx, model->key, (uint8_t)MNV_CHACHA20_KEY_SIZE);
#if defined(MNV_PROGMEM_WEIGHTS)
        {
            uint8_t chunk[64];
            uint16_t remaining = model->encrypted_len;
            uint16_t offset    = 0;
            while (remaining > 0) {
                uint16_t n = (remaining > 64U) ? 64U : remaining;
                for (uint16_t i = 0; i < n; i++)
                    chunk[i] = pgm_read_byte(model->encrypted_weights + offset + i);
                mnv_blake2s_update(&bctx, chunk, n);
                offset    += n;
                remaining -= n;
            }
            mnv_secure_zero(chunk, sizeof(chunk));
        }
#else
        mnv_blake2s_update(&bctx, model->encrypted_weights, model->encrypted_len);
#endif
        uint8_t computed_mac[MNV_BLAKE2S_DIGEST_SIZE];
        mnv_blake2s_final(&bctx, computed_mac);
        uint8_t diff = mnv_ct_compare(computed_mac,
                                       model->crypto->mac,
                                       MNV_BLAKE2S_DIGEST_SIZE);
        mnv_secure_zero(computed_mac, sizeof(computed_mac));
        s = (diff == 0) ? MNV_OK : MNV_ERR_TAMPER;
    }
    if (s != MNV_OK) { mnv_secure_zero(ctx, sizeof(mnv_ctx_t)); return MNV_ERR_TAMPER; }

    ctx->verified    = true;
    ctx->initialized = true;
    return MNV_OK;
}

void mnv_destroy(mnv_ctx_t *ctx)
{
    if (ctx) mnv_secure_zero(ctx, sizeof(mnv_ctx_t));
}

/* ── v1.1: PRNG seeding ─────────────────────────────────────────────────── */

void mnv_seed_prng(mnv_ctx_t *ctx, uint32_t seed)
{
    if (!ctx) return;
    ctx->prng_state = (seed != 0U) ? seed : 0xDEADC0DEUL;
}

/* ── Inference ──────────────────────────────────────────────────────────── */

mnv_status_t mnv_run_with_model(mnv_ctx_t         *ctx,
                                 const mnv_model_t *model,
                                 const mnv_act_t   *input,
                                 mnv_act_t         *output)
{
    if (!ctx || !model || !input || !output) return MNV_ERR_NULL;
    if (!ctx->initialized)                   return MNV_ERR_CONFIG;
    if (!ctx->verified)                      return MNV_ERR_TAMPER;

    mnv_status_t status;

    /* Pre-inference canary */
    status = mnv_canary_check(ctx);
    if (status != MNV_OK) goto fail_glitch;

    /* Input validation */
#if defined(MNV_ENABLE_INPUT_VALIDATION)
    status = mnv_ct_validate_input(input, MNV_INPUT_SIZE);
    if (status != MNV_OK) { mnv_secure_zero(output, MNV_OUTPUT_SIZE); return MNV_ERR_INPUT; }
#endif

    /* Run 1 */
    {
        mnv_chacha20_ctx_t chacha;
        engine_chacha_init(&chacha, model);
        status = engine_forward(ctx, model, input, output, &chacha);
        mnv_chacha20_wipe(&chacha);
        if (status != MNV_OK) goto fail;
    }

    /* Run 2 — double-run comparison */
#if defined(MNV_ENABLE_DOUBLE_RUN)
    {
        mnv_chacha20_ctx_t chacha2;
        engine_chacha_init(&chacha2, model);
        status = engine_forward(ctx, model, input, ctx->run2_buf, &chacha2);
        mnv_chacha20_wipe(&chacha2);
        if (status != MNV_OK) goto fail;
        uint8_t diff = mnv_ct_compare((const uint8_t *)output,
                                       (const uint8_t *)ctx->run2_buf,
                                       MNV_OUTPUT_SIZE);
        mnv_secure_zero(ctx->run2_buf, MNV_OUTPUT_SIZE);
        if (diff != 0U) { status = MNV_ERR_MISMATCH; goto fail; }
    }
#endif

    /* Post-inference canary */
    status = mnv_canary_check(ctx);
    if (status != MNV_OK) goto fail_glitch;

    /* Confidence check */
#if defined(MNV_ENABLE_INPUT_VALIDATION)
    status = mnv_ct_confidence_check(output, MNV_OUTPUT_SIZE);
    if (status != MNV_OK) { mnv_secure_zero(output, MNV_OUTPUT_SIZE); return MNV_ERR_CONFIDENCE; }
#endif

    /* v1.1: Output MAC */
#if defined(MNV_ENABLE_OUTPUT_AUTH)
    mnv_outauth_compute(ctx, model->key, output, input);
#endif

    return MNV_OK;

fail_glitch:
    status = MNV_ERR_GLITCH;
fail:
    mnv_secure_zero(output,              MNV_OUTPUT_SIZE);
    mnv_secure_zero(ctx->weight_scratch, sizeof(ctx->weight_scratch));
    mnv_secure_zero(ctx->buf_a,          sizeof(ctx->buf_a));
    mnv_secure_zero(ctx->buf_b,          sizeof(ctx->buf_b));
    mnv_secure_zero(ctx->output_mac,     MNV_OUTPUT_MAC_SIZE);
    return status;
}

/* Shim: mnv_run() without model pointer cannot function */
mnv_status_t mnv_run(mnv_ctx_t *ctx, const mnv_act_t *input, mnv_act_t *output)
{
    (void)ctx; (void)input; (void)output;
    return MNV_ERR_CONFIG;
}

/* ── Verification ──────────────────────────────────────────────────────── */

mnv_status_t mnv_verify(mnv_ctx_t *ctx, const mnv_model_t *model)
{
    if (!ctx || !model) return MNV_ERR_NULL;
    mnv_status_t s = mnv_blake2s_verify(
        model->key, (uint8_t)MNV_CHACHA20_KEY_SIZE,
        model->encrypted_weights, model->encrypted_len,
        model->crypto->mac);
    ctx->verified = (s == MNV_OK);
    return s;
}

/* ── v1.1: Output auth API ─────────────────────────────────────────────── */

mnv_status_t mnv_verify_output_with_key(const mnv_ctx_t *ctx,
                                         const uint8_t   *device_key,
                                         const mnv_act_t *input,
                                         const mnv_act_t *output)
{
    if (!ctx || !device_key || !input || !output) return MNV_ERR_NULL;
    return mnv_outauth_verify(ctx, device_key, output, input);
}

void mnv_get_output_mac(const mnv_ctx_t *ctx, uint8_t *mac)
{
    if (!ctx || !mac) return;
    memcpy(mac, ctx->output_mac, MNV_OUTPUT_MAC_SIZE);
}

/* mnv_secure_zero and mnv_ct_compare defined in mnv_ct.c */
