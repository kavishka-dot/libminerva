/**
 * @file mnv_chacha20.c
 * @brief ChaCha20 stream cipher — constant-time weight decryption
 *
 * Implementation of RFC 7539 ChaCha20.
 * Properties relevant to Minerva:
 *   - No S-box: immune to cache-timing attacks (AVR has no cache, but principle holds)
 *   - Arithmetic only: ADD, XOR, ROTATE — constant-time on all targets
 *   - ~3 KB flash on AVR
 *   - No dynamic allocation: state is caller-provided
 *
 * Usage: decrypt one layer's worth of weights at a time.
 * The weight_scratch buffer is zeroed immediately after the layer forward pass.
 */

#include "mnv_chacha20.h"
#include <string.h>

/* =========================================================================
 * INTERNAL MACROS
 * ========================================================================= */

#define ROTL32(x, n)  (((x) << (n)) | ((x) >> (32u - (n))))

#define QR(a, b, c, d)          \
    (a) += (b); (d) ^= (a); (d) = ROTL32((d), 16); \
    (c) += (d); (b) ^= (c); (b) = ROTL32((b), 12); \
    (a) += (b); (d) ^= (a); (d) = ROTL32((d),  8); \
    (c) += (d); (b) ^= (c); (b) = ROTL32((b),  7);

/* Little-endian load/store — portable, no UB */
static inline uint32_t load32_le(const uint8_t *p)
{
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline void store32_le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/* =========================================================================
 * CORE BLOCK FUNCTION
 * ========================================================================= */

/**
 * @brief Generate one 64-byte ChaCha20 keystream block.
 *
 * @param key      32-byte key
 * @param nonce    12-byte nonce (RFC 7539)
 * @param counter  32-bit block counter
 * @param out      64-byte output block
 */
void mnv_chacha20_block(const uint8_t *key,
                        const uint8_t *nonce,
                        uint32_t       counter,
                        uint8_t       *out)
{
    uint32_t s[16];

    /* Initialize state — ChaCha20 constants "expand 32-byte k" */
    s[0]  = 0x61707865UL;
    s[1]  = 0x3320646eUL;
    s[2]  = 0x79622d32UL;
    s[3]  = 0x6b206574UL;
    s[4]  = load32_le(key +  0);
    s[5]  = load32_le(key +  4);
    s[6]  = load32_le(key +  8);
    s[7]  = load32_le(key + 12);
    s[8]  = load32_le(key + 16);
    s[9]  = load32_le(key + 20);
    s[10] = load32_le(key + 24);
    s[11] = load32_le(key + 28);
    s[12] = counter;
    s[13] = load32_le(nonce + 0);
    s[14] = load32_le(nonce + 4);
    s[15] = load32_le(nonce + 8);

    uint32_t x[16];
    for (int i = 0; i < 16; i++) x[i] = s[i];

    /* 20 rounds (10 column + 10 diagonal) */
    for (int i = 0; i < 10; i++) {
        /* Column rounds */
        QR(x[0], x[4], x[ 8], x[12]);
        QR(x[1], x[5], x[ 9], x[13]);
        QR(x[2], x[6], x[10], x[14]);
        QR(x[3], x[7], x[11], x[15]);
        /* Diagonal rounds */
        QR(x[0], x[5], x[10], x[15]);
        QR(x[1], x[6], x[11], x[12]);
        QR(x[2], x[7], x[ 8], x[13]);
        QR(x[3], x[4], x[ 9], x[14]);
    }

    for (int i = 0; i < 16; i++) x[i] += s[i];
    for (int i = 0; i < 16; i++) store32_le(out + 4 * i, x[i]);
}

/* =========================================================================
 * STREAM DECRYPTION
 * ========================================================================= */

/**
 * @brief Initialize ChaCha20 decryption context.
 */
void mnv_chacha20_init(mnv_chacha20_ctx_t *ctx,
                       const uint8_t      *key,
                       const uint8_t      *nonce,
                       uint32_t            initial_counter)
{
    memcpy(ctx->key,   key,   MNV_CHACHA20_KEY_SIZE);
    memcpy(ctx->nonce, nonce, MNV_CHACHA20_IV_SIZE);
    ctx->counter      = initial_counter;
    ctx->block_pos    = 64; /* force block generation on first use */
    /* Zero block buffer */
    memset(ctx->block, 0, sizeof(ctx->block));
}

/**
 * @brief Decrypt (XOR) len bytes from ciphertext into plaintext.
 *
 * Processes one 64-byte block at a time. Suitable for streaming
 * layer-by-layer weight decryption.
 */
void mnv_chacha20_decrypt(mnv_chacha20_ctx_t *ctx,
                          const uint8_t      *ciphertext,
                          uint8_t            *plaintext,
                          uint16_t            len)
{
    for (uint16_t i = 0; i < len; i++) {
        if (ctx->block_pos >= 64) {
            mnv_chacha20_block(ctx->key, ctx->nonce, ctx->counter, ctx->block);
            ctx->counter++;
            ctx->block_pos = 0;
        }
#if defined(MNV_PROGMEM_WEIGHTS)
        plaintext[i] = pgm_read_byte(ciphertext + i) ^ ctx->block[ctx->block_pos++];
#else
        plaintext[i] = ciphertext[i] ^ ctx->block[ctx->block_pos++];
#endif
    }
}

/**
 * @brief Securely wipe ChaCha20 context (key material).
 */
void mnv_chacha20_wipe(mnv_chacha20_ctx_t *ctx)
{
    volatile uint8_t *p = (volatile uint8_t *)ctx;
    for (size_t i = 0; i < sizeof(mnv_chacha20_ctx_t); i++) p[i] = 0;
}
