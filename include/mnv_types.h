/**
 * @file mnv_types.h
 * @brief Minerva type definitions, status codes, and model descriptor
 *
 * All types used throughout Minerva are defined here.
 * Fixed-width integers are used exclusively — no int, no long.
 */

#ifndef MNV_TYPES_H
#define MNV_TYPES_H

#include "mnv_config.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#if defined(MNV_ARCH_AVR8) && !defined(MNV_TARGET_HOST)
    #include <avr/pgmspace.h>
#else
    /* Host / non-AVR: provide no-op PROGMEM stub */
    #ifndef PROGMEM
    #define PROGMEM
    #endif
    #ifndef pgm_read_byte
    #define pgm_read_byte(p) (*(p))
    #endif
    #ifndef memcpy_P
    #include <string.h>
    #define memcpy_P(d,s,n) memcpy((d),(s),(n))
    #endif
#endif

/* =========================================================================
 * COMPILE-TIME ASSERTIONS
 * Enforced before a single byte of inference runs.
 * ========================================================================= */

#define MNV_STATIC_ASSERT(cond, msg) \
    typedef char mnv_static_assert_##msg[(cond) ? 1 : -1]

/* Weight budget check */
MNV_STATIC_ASSERT(
    MNV_TOTAL_WEIGHT_COUNT + MNV_TOTAL_BIAS_COUNT <= MNV_NET_WEIGHT_BUDGET,
    weight_budget_exceeded
);

/* SRAM budget: largest layer activation buffer must fit */
MNV_STATIC_ASSERT(
    MNV_LAYER_0_SIZE <= MNV_MAX_SRAM_BUDGET,
    sram_budget_exceeded
);

/* Q15 only on targets with enough RAM */
#if defined(MNV_QUANT_Q15)
MNV_STATIC_ASSERT(
    MNV_SRAM_BYTES >= 4096U,
    q15_requires_at_least_4kb_sram
);
#endif

/* =========================================================================
 * QUANTIZED TYPES
 * ========================================================================= */

#if defined(MNV_QUANT_Q8)
    typedef int8_t   mnv_weight_t;    /* signed 8-bit weight                        */
    typedef int8_t   mnv_act_t;       /* signed 8-bit activation                    */
    typedef int32_t  mnv_acc_t;       /* 32-bit accumulator — MUST be 32-bit.       */
                                      /* int16 overflows for layers > 2 inputs:     */
                                      /* N * 127 * 127 > 32767 for N >= 3.          */
                                      /* On AVR, int32 arithmetic is emulated but   */
                                      /* costs only ~18us extra per inference.       */
    typedef int8_t   mnv_bias_t;      /* signed 8-bit bias                          */
    #define MNV_Q_SCALE     127       /* Q8 full-scale                              */
    #define MNV_Q_MIN      -128
    #define MNV_Q_MAX       127

#elif defined(MNV_QUANT_Q4)
    typedef int8_t   mnv_weight_t;    /* packed 2x per byte, unpacked to int8 */
    typedef int8_t   mnv_act_t;
    typedef int32_t  mnv_acc_t;  /* 32-bit for safety */
    typedef int8_t   mnv_bias_t;
    #define MNV_Q_SCALE     7
    #define MNV_Q_MIN      -8
    #define MNV_Q_MAX       7

#elif defined(MNV_QUANT_Q15)
    typedef int16_t  mnv_weight_t;
    typedef int16_t  mnv_act_t;
    typedef int32_t  mnv_acc_t;
    typedef int16_t  mnv_bias_t;
    #define MNV_Q_SCALE     32767
    #define MNV_Q_MIN      -32768
    #define MNV_Q_MAX       32767

#elif defined(MNV_QUANT_BINARY)
    typedef uint8_t  mnv_weight_t;    /* packed bits: 8 weights per byte  */
    typedef int8_t   mnv_act_t;
    typedef int16_t  mnv_acc_t;
    typedef int8_t   mnv_bias_t;
    #define MNV_Q_SCALE     1
    #define MNV_Q_MIN      -1
    #define MNV_Q_MAX       1
#endif

/* =========================================================================
 * STATUS CODES
 * Every public function returns mnv_status_t.
 * On any non-OK status the engine zeroes its SRAM buffers immediately.
 * ========================================================================= */

typedef enum {
    MNV_OK              = 0x00,  /* Success                                   */
    MNV_ERR_TAMPER      = 0x01,  /* Integrity check failed — model modified    */
    MNV_ERR_GLITCH      = 0x02,  /* Canary corrupted — fault injection likely  */
    MNV_ERR_INPUT       = 0x03,  /* Input out of valid range                  */
    MNV_ERR_CONFIDENCE  = 0x04,  /* Output confidence below MNV_MIN_CONFIDENCE */
    MNV_ERR_CONFIG      = 0x05,  /* Bad compile-time configuration            */
    MNV_ERR_DECRYPT     = 0x06,  /* Decryption failure                        */
    MNV_ERR_MISMATCH    = 0x07,  /* Double-run results diverged — glitch       */
    MNV_ERR_NULL        = 0x08,  /* Null pointer argument                     */
} mnv_status_t;

/* =========================================================================
 * ACTIVATION FUNCTION SELECTOR
 * ========================================================================= */

typedef enum {
    MNV_ACT_RELU    = 0,
    MNV_ACT_SIGMOID = 1,
    MNV_ACT_TANH    = 2,
    MNV_ACT_LINEAR  = 3,   /* output layer only */
    MNV_ACT_SIGN    = 4,   /* BNN only */
} mnv_act_fn_t;

/* =========================================================================
 * LAYER DESCRIPTOR
 * Describes one layer of the model at compile time.
 * All fields are const — never modified at runtime.
 * ========================================================================= */

typedef struct {
    uint16_t      input_size;
    uint16_t      output_size;
    mnv_act_fn_t  activation;
    /* Pointers into the weight/bias arrays in flash (PROGMEM on AVR) */
#if defined(MNV_PROGMEM_WEIGHTS)
    const mnv_weight_t * PROGMEM weights;
    const mnv_bias_t   * PROGMEM biases;
#else
    const mnv_weight_t *weights;
    const mnv_bias_t   *biases;
#endif
} mnv_layer_desc_t;

/* =========================================================================
 * CNN1D LAYER DESCRIPTOR
 * ========================================================================= */

typedef struct {
    uint16_t      input_len;
    uint16_t      num_filters;
    uint16_t      kernel_size;
    uint16_t      pool_size;
    mnv_act_fn_t  activation;
#if defined(MNV_PROGMEM_WEIGHTS)
    const mnv_weight_t * PROGMEM kernels;
    const mnv_bias_t   * PROGMEM biases;
#else
    const mnv_weight_t *kernels;
    const mnv_bias_t   *biases;
#endif
} mnv_conv1d_desc_t;

/* =========================================================================
 * CRYPTO HEADER
 * Prepended to every encrypted weight blob.
 * ========================================================================= */

typedef struct {
    uint8_t  iv[MNV_CHACHA20_IV_SIZE];      /* ChaCha20 nonce            */
    uint8_t  mac[MNV_BLAKE2S_DIGEST_SIZE];  /* BLAKE2s over plaintext    */
    uint16_t weight_count;                  /* number of weights         */
    uint16_t bias_count;                    /* number of biases          */
} mnv_crypto_header_t;

/* =========================================================================
 * MODEL DESCRIPTOR
 * The single struct the user passes to mnv_init().
 * Typically lives in flash (PROGMEM on AVR).
 * ========================================================================= */

typedef struct {
    uint8_t             version;            /* must equal MNV_ABI_VERSION        */
    uint8_t             num_layers;
    const mnv_layer_desc_t *layers;         /* array of layer descriptors        */
    const mnv_crypto_header_t *crypto;      /* crypto header for weight blob     */
    const uint8_t       *key;               /* 256-bit ChaCha20 key (from fuse/KDF)*/
    const uint8_t       *encrypted_weights; /* entire encrypted weight blob      */
    uint16_t             encrypted_len;
} mnv_model_t;

#define MNV_ABI_VERSION  0x01U

/* =========================================================================
 * INFERENCE CONTEXT
 * Statically allocated workspace. One instance per application.
 * Zero malloc.
 * ========================================================================= */

typedef struct {
    /* Activation buffers. For MLP: sized to widest layer (MNV_LAYER_0_SIZE).
     * For CNN1D: buf_a holds the flattened feature map (MNV_CNN_FLAT_SIZE).
     * MNV_CTX_BUF_SIZE is the maximum of the two. */
#if defined(MNV_ARCH_CNN1D)
#  define MNV_CTX_BUF_SIZE  (MNV_CNN_NUM_FILTERS * ((MNV_INPUT_SIZE - MNV_CNN_KERNEL_SIZE + 1U) / MNV_CNN_POOL_SIZE))
#elif defined(MNV_LAYER_0_SIZE)
#  define MNV_CTX_BUF_SIZE  MNV_LAYER_0_SIZE
#else
#  define MNV_CTX_BUF_SIZE  32U
#endif
    mnv_act_t   buf_a[MNV_CTX_BUF_SIZE];
    mnv_act_t   buf_b[MNV_CTX_BUF_SIZE];

    /* Decryption scratch — one layer of weights at a time */
    /* Weight scratch: for MLP = widest_layer * input_size.
     * For CNN1D = output_size * flat_size (dense layer weights). */
    /* For CNN1D: holds one dense output row (flat_size weights).
     * For MLP: holds one full layer (widest_layer * input_size). */
/* CNN1D scratch must hold dense layer weights: OUTPUT_SIZE * FLAT_SIZE
 * FLAT_SIZE = CNN_NUM_FILTERS * POOL_LEN = CNN_NUM_FILTERS * ((INPUT_SIZE-KERNEL_SIZE+1)/POOL_SIZE) */
#if defined(MNV_ARCH_CNN1D)
#  define MNV_CNN_FLAT_SZ (MNV_CNN_NUM_FILTERS * ((MNV_INPUT_SIZE - MNV_CNN_KERNEL_SIZE + 1U) / MNV_CNN_POOL_SIZE))
    mnv_weight_t weight_scratch[MNV_OUTPUT_SIZE * MNV_CNN_FLAT_SZ];
#else
    mnv_weight_t weight_scratch[MNV_LAYER_0_SIZE * MNV_INPUT_SIZE];
#endif

    /* Double-run comparison buffer */
    mnv_act_t   run2_buf[MNV_OUTPUT_SIZE];

    /* ChaCha20 keystream block */
    uint8_t     chacha_block[64];
    uint32_t    chacha_counter;

    /* Canary sentinels — checked after every layer */
    uint32_t    canary_pre[MNV_CANARY_COUNT];
    uint32_t    canary_post[MNV_CANARY_COUNT];

    /* v1.1: Output authentication MAC (appended after inference) */
    uint8_t     output_mac[MNV_OUTPUT_MAC_SIZE];

    /* v1.1: Monotonic inference counter for replay prevention */
    uint32_t    inference_counter;

    /* v1.1: Xorshift32 PRNG state for LUT blinding */
    uint32_t    prng_state;

    /* State flags */
    bool        verified;     /* integrity check passed this session */
    bool        initialized;
} mnv_ctx_t;

#endif /* MNV_TYPES_H */
