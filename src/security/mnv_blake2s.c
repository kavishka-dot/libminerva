/**
 * @file mnv_blake2s.c
 * @brief BLAKE2s-256 keyed MAC — fixed rotation direction (v1.1 bugfix)
 *
 * BLAKE2s uses ROTATE RIGHT in its G mixing function.
 * The v1.0 implementation incorrectly used ROTATE LEFT for the 12/8/7-bit
 * rotations. This is corrected here.
 */

#include "mnv_blake2s.h"
#include "mnv_ct.h"
#include <string.h>

/* ── Constants ─────────────────────────────────────────────────────────── */

static const uint32_t BLAKE2S_IV[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

static const uint8_t BLAKE2S_SIGMA[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
};

/* BLAKE2s uses ROTATE RIGHT */
#define ROTR32(x, n) (((uint32_t)(x) >> (n)) | ((uint32_t)(x) << (32U - (n))))

/* G mixing function — RFC 7693 §3.1 */
#define B2S_G(v, a, b, c, d, x, y)              \
    (v)[a] += (v)[b] + (x);                     \
    (v)[d]  = ROTR32((v)[d] ^ (v)[a], 16);      \
    (v)[c] += (v)[d];                            \
    (v)[b]  = ROTR32((v)[b] ^ (v)[c], 12);      \
    (v)[a] += (v)[b] + (y);                      \
    (v)[d]  = ROTR32((v)[d] ^ (v)[a],  8);      \
    (v)[c] += (v)[d];                            \
    (v)[b]  = ROTR32((v)[b] ^ (v)[c],  7);

static inline uint32_t load32_le(const uint8_t *p)
{
    return (uint32_t)p[0]        | ((uint32_t)p[1] <<  8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void store32_le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;       p[1] = (uint8_t)(v >>  8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

/* ── Compress ───────────────────────────────────────────────────────────── */

static void blake2s_compress(mnv_blake2s_ctx_t *ctx, const uint8_t *block)
{
    uint32_t m[16], v[16];
    for (int i = 0; i < 16; i++) m[i] = load32_le(block + 4 * i);
    for (int i = 0; i <  8; i++) v[i] = ctx->h[i];

    v[ 8] = BLAKE2S_IV[0];
    v[ 9] = BLAKE2S_IV[1];
    v[10] = BLAKE2S_IV[2];
    v[11] = BLAKE2S_IV[3];
    v[12] = BLAKE2S_IV[4] ^ ctx->t[0];
    v[13] = BLAKE2S_IV[5] ^ ctx->t[1];
    v[14] = BLAKE2S_IV[6] ^ ctx->f[0];
    v[15] = BLAKE2S_IV[7] ^ ctx->f[1];

    for (int r = 0; r < 10; r++) {
        const uint8_t *s = BLAKE2S_SIGMA[r];
        B2S_G(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        B2S_G(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        B2S_G(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        B2S_G(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        B2S_G(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        B2S_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        B2S_G(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        B2S_G(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) ctx->h[i] ^= v[i] ^ v[i + 8];
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void mnv_blake2s_init(mnv_blake2s_ctx_t *ctx,
                      const uint8_t     *key,
                      uint8_t            keylen)
{
    memset(ctx, 0, sizeof(*ctx));
    for (int i = 0; i < 8; i++) ctx->h[i] = BLAKE2S_IV[i];
    ctx->h[0] ^= 0x01010000UL | ((uint32_t)keylen << 8) | 32U;
    ctx->outlen = 32;

    if (keylen > 0) {
        uint8_t block[64];
        memset(block, 0, 64);
        memcpy(block, key, keylen);
        mnv_blake2s_update(ctx, block, 64);
        volatile uint8_t *vp = (volatile uint8_t *)block;
        for (int i = 0; i < 64; i++) vp[i] = 0;
    }
}

void mnv_blake2s_update(mnv_blake2s_ctx_t *ctx,
                        const uint8_t     *data,
                        uint16_t           len)
{
    while (len > 0) {
        uint16_t fill = (uint16_t)(64 - ctx->buflen);
        if (fill > len) fill = len;
        memcpy(ctx->buf + ctx->buflen, data, fill);
        ctx->buflen += fill;
        data += fill;
        len  -= fill;

        if (ctx->buflen == 64 && len > 0) {
            ctx->t[0] += 64U;
            if (ctx->t[0] < 64U) ctx->t[1]++;
            blake2s_compress(ctx, ctx->buf);
            ctx->buflen = 0;
        }
    }
}

void mnv_blake2s_final(mnv_blake2s_ctx_t *ctx, uint8_t *digest)
{
    ctx->t[0] += ctx->buflen;
    if (ctx->t[0] < ctx->buflen) ctx->t[1]++;
    ctx->f[0] = 0xFFFFFFFFUL;

    memset(ctx->buf + ctx->buflen, 0, (size_t)(64 - ctx->buflen));
    blake2s_compress(ctx, ctx->buf);

    for (int i = 0; i < 8; i++) store32_le(digest + 4 * i, ctx->h[i]);

    volatile uint32_t *vp = (volatile uint32_t *)ctx->h;
    for (int i = 0; i < 8; i++) vp[i] = 0;
}

void mnv_blake2s_mac(const uint8_t *key,    uint8_t  keylen,
                     const uint8_t *data,   uint16_t datalen,
                     uint8_t       *digest)
{
    mnv_blake2s_ctx_t ctx;
    mnv_blake2s_init(&ctx, key, keylen);
    mnv_blake2s_update(&ctx, data, datalen);
    mnv_blake2s_final(&ctx, digest);
}

mnv_status_t mnv_blake2s_verify(const uint8_t *key,    uint8_t  keylen,
                                const uint8_t *data,   uint16_t datalen,
                                const uint8_t *expected_mac)
{
    uint8_t computed[MNV_BLAKE2S_DIGEST_SIZE];
    mnv_blake2s_mac(key, keylen, data, datalen, computed);
    uint8_t diff = mnv_ct_compare(computed, expected_mac, MNV_BLAKE2S_DIGEST_SIZE);
    volatile uint8_t *vp = (volatile uint8_t *)computed;
    for (int i = 0; i < MNV_BLAKE2S_DIGEST_SIZE; i++) vp[i] = 0;
    return (diff == 0) ? MNV_OK : MNV_ERR_TAMPER;
}
