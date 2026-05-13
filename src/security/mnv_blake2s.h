/**
 * @file mnv_blake2s.h
 * @brief BLAKE2s integrity verification — internal header
 */

#ifndef MNV_BLAKE2S_H
#define MNV_BLAKE2S_H

#include "mnv_types.h"

typedef struct {
    uint32_t h[8];
    uint32_t t[2];
    uint32_t f[2];
    uint8_t  buf[64];
    uint8_t  buflen;
    uint8_t  outlen;
} mnv_blake2s_ctx_t;

void         mnv_blake2s_init(mnv_blake2s_ctx_t *ctx,
                              const uint8_t *key, uint8_t keylen);
void         mnv_blake2s_update(mnv_blake2s_ctx_t *ctx,
                                const uint8_t *data, uint16_t len);
void         mnv_blake2s_final(mnv_blake2s_ctx_t *ctx, uint8_t *digest);
void         mnv_blake2s_mac(const uint8_t *key,  uint8_t  keylen,
                             const uint8_t *data, uint16_t datalen,
                             uint8_t *digest);
mnv_status_t mnv_blake2s_verify(const uint8_t *key,  uint8_t  keylen,
                                const uint8_t *data, uint16_t datalen,
                                const uint8_t *expected_mac);

#endif /* MNV_BLAKE2S_H */
