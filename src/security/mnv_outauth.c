/**
 * @file mnv_outauth.c
 * @brief Output authentication — session MAC over inference results (v1.1)
 *
 * PROBLEM:
 *   In v1.0, inference output is produced and passed to the application
 *   without any integrity guarantee on the output bus. An adversary with
 *   access to the SPI/UART/I²C line between the MCU and an actuator can:
 *     - Intercept and replace the inference result
 *     - Replay a previously valid result at a different time
 *
 * SOLUTION (v1.1): Per-inference session MAC
 *   After every successful inference, Minerva computes:
 *
 *     MAC = BLAKE2s(
 *       key  = device_key,
 *       data = output[0..OUTPUT_SIZE]
 *              || input[0..INPUT_SIZE]
 *              || counter[0..3]          ← monotonic inference counter
 *     )
 *
 *   The MAC is truncated to MNV_OUTPUT_MAC_SIZE (8) bytes.
 *
 *   The downstream consumer (e.g. a separate MCU or a gateway) holds the
 *   same device key and counter state. It verifies the MAC before acting
 *   on the inference result.
 *
 *   Properties:
 *     - Replay prevention: counter makes each MAC unique
 *     - Integrity: any modification to output or input invalidates MAC
 *     - Authenticity: only the device holding the key can produce valid MACs
 *     - Compact: 8 bytes overhead per inference
 *
 *   Limitation: counter is in SRAM — resets on power cycle. For persistent
 *   counter, write to EEPROM every N inferences (v1.2 roadmap).
 */

#include "mnv_outauth.h"
#include "mnv_blake2s.h"
#include "mnv_ct.h"
#include <string.h>

/* =========================================================================
 * INTERNAL: BUILD MAC INPUT BUFFER
 * Layout: [output | input | counter_le32]
 * Total:  OUTPUT_SIZE + INPUT_SIZE + 4 bytes
 * ========================================================================= */

#define MNV_AUTH_BUF_SIZE  (MNV_OUTPUT_SIZE + MNV_INPUT_SIZE + 4U)

static void build_auth_buffer(const mnv_act_t *output,
                               const mnv_act_t *input,
                               uint32_t         counter,
                               uint8_t         *buf)
{
    /* Output vector */
    memcpy(buf, output, MNV_OUTPUT_SIZE);
    /* Input vector */
    memcpy(buf + MNV_OUTPUT_SIZE, input, MNV_INPUT_SIZE);
    /* Counter — little-endian */
    buf[MNV_OUTPUT_SIZE + MNV_INPUT_SIZE + 0] = (uint8_t)(counter);
    buf[MNV_OUTPUT_SIZE + MNV_INPUT_SIZE + 1] = (uint8_t)(counter >> 8);
    buf[MNV_OUTPUT_SIZE + MNV_INPUT_SIZE + 2] = (uint8_t)(counter >> 16);
    buf[MNV_OUTPUT_SIZE + MNV_INPUT_SIZE + 3] = (uint8_t)(counter >> 24);
}

/* =========================================================================
 * PUBLIC API
 * ========================================================================= */

/**
 * @brief Compute output MAC and store in ctx->output_mac.
 *
 * Called internally by mnv_run_with_model() after successful inference.
 * Increments ctx->inference_counter.
 */
void mnv_outauth_compute(mnv_ctx_t       *ctx,
                          const uint8_t   *device_key,
                          const mnv_act_t *output,
                          const mnv_act_t *input)
{
    uint8_t auth_buf[MNV_AUTH_BUF_SIZE];
    uint8_t full_mac[MNV_BLAKE2S_DIGEST_SIZE];

    build_auth_buffer(output, input, ctx->inference_counter, auth_buf);
    mnv_blake2s_mac(device_key, (uint8_t)MNV_CHACHA20_KEY_SIZE,
                    auth_buf, (uint16_t)MNV_AUTH_BUF_SIZE,
                    full_mac);

    /* Truncate to MNV_OUTPUT_MAC_SIZE bytes */
    memcpy(ctx->output_mac, full_mac, MNV_OUTPUT_MAC_SIZE);

    /* Increment counter — monotonic, wraps at 2^32 */
    ctx->inference_counter++;

    /* Wipe temporaries */
    mnv_secure_zero(auth_buf,  sizeof(auth_buf));
    mnv_secure_zero(full_mac,  sizeof(full_mac));
}

/**
 * @brief Verify the MAC stored in ctx against the given output and input.
 *
 * The counter used is (ctx->inference_counter - 1) — the counter value
 * at the time the last MAC was computed.
 *
 * @return MNV_OK if valid, MNV_ERR_TAMPER if invalid.
 */
mnv_status_t mnv_outauth_verify(const mnv_ctx_t *ctx,
                                 const uint8_t   *device_key,
                                 const mnv_act_t *output,
                                 const mnv_act_t *input)
{
    uint8_t auth_buf[MNV_AUTH_BUF_SIZE];
    uint8_t full_mac[MNV_BLAKE2S_DIGEST_SIZE];

    /* Use the counter value at the time of the last inference */
    uint32_t last_counter = ctx->inference_counter - 1U;

    build_auth_buffer(output, input, last_counter, auth_buf);
    mnv_blake2s_mac(device_key, (uint8_t)MNV_CHACHA20_KEY_SIZE,
                    auth_buf, (uint16_t)MNV_AUTH_BUF_SIZE,
                    full_mac);

    /* Constant-time compare — truncated MAC only */
    uint8_t diff = mnv_ct_compare(full_mac, ctx->output_mac, MNV_OUTPUT_MAC_SIZE);

    mnv_secure_zero(auth_buf, sizeof(auth_buf));
    mnv_secure_zero(full_mac, sizeof(full_mac));

    return (diff == 0) ? MNV_OK : MNV_ERR_TAMPER;
}
