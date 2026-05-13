/**
 * @file mnv_chacha20.h
 * @brief ChaCha20 stream cipher — internal header
 */

#ifndef MNV_CHACHA20_H
#define MNV_CHACHA20_H

#include "mnv_types.h"

typedef struct {
    uint8_t  key[MNV_CHACHA20_KEY_SIZE];
    uint8_t  nonce[MNV_CHACHA20_IV_SIZE];
    uint32_t counter;
    uint8_t  block[64];
    uint8_t  block_pos;
} mnv_chacha20_ctx_t;

void mnv_chacha20_block(const uint8_t *key, const uint8_t *nonce,
                        uint32_t counter, uint8_t *out);
void mnv_chacha20_init(mnv_chacha20_ctx_t *ctx, const uint8_t *key,
                       const uint8_t *nonce, uint32_t initial_counter);
void mnv_chacha20_decrypt(mnv_chacha20_ctx_t *ctx, const uint8_t *ciphertext,
                          uint8_t *plaintext, uint16_t len);
void mnv_chacha20_wipe(mnv_chacha20_ctx_t *ctx);

#endif /* MNV_CHACHA20_H */
