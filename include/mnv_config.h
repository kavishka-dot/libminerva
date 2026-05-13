/**
 * @file mnv_config.h
 * @brief Minerva compile-time configuration
 *
 * MINERVA — Minimal Inference Engine for Robust, Verifiable, and Authenticated ML
 *
 * Edit this file to configure Minerva for your target MCU and model.
 * All memory is statically allocated based on these values.
 * No allocation occurs at runtime — ever.
 *
 * The Three Minerva Laws:
 *   I.   Minerva never produces output from an unverified model.
 *   II.  Minerva reveals nothing through timing, power, or output behavior.
 *   III. Minerva never allocates. Every byte is known at compile time.
 */

#ifndef MNV_CONFIG_H
#define MNV_CONFIG_H

/* =========================================================================
 * TARGET SELECTION
 * Uncomment exactly one target.
 * ========================================================================= */

#define MNV_TARGET_ATMEGA328P
/* #define MNV_TARGET_ATMEGA2560   */
/* #define MNV_TARGET_ATTINY85     */
/* #define MNV_TARGET_STM32F0      */
/* #define MNV_TARGET_STM32F4      */
/* #define MNV_TARGET_HOST         */   /* PC/Linux — for testing only */

/* =========================================================================
 * TARGET-DERIVED CONSTRAINTS
 * These are enforced via static_assert in mnv_types.h.
 * ========================================================================= */

#if defined(MNV_TARGET_ATMEGA328P)
    #define MNV_FLASH_BYTES         32768U
    #define MNV_SRAM_BYTES          2048U
    #define MNV_MAX_WEIGHT_BYTES    14336U   /* ~14 KB after code + crypto overhead */
    #define MNV_MAX_SRAM_BUDGET     960U     /* bytes available for inference buffers */
    #define MNV_ARCH_AVR8
    #define MNV_NO_FPU
    #define MNV_PROGMEM_WEIGHTS              /* weights live in flash via PROGMEM */

#elif defined(MNV_TARGET_ATMEGA2560)
    #define MNV_FLASH_BYTES         262144U
    #define MNV_SRAM_BYTES          8192U
    #define MNV_MAX_WEIGHT_BYTES    204800U
    #define MNV_MAX_SRAM_BUDGET     6144U
    #define MNV_ARCH_AVR8
    #define MNV_NO_FPU
    #define MNV_PROGMEM_WEIGHTS

#elif defined(MNV_TARGET_ATTINY85)
    #define MNV_FLASH_BYTES         8192U
    #define MNV_SRAM_BYTES          512U
    #define MNV_MAX_WEIGHT_BYTES    2048U
    #define MNV_MAX_SRAM_BUDGET     384U
    #define MNV_ARCH_AVR8
    #define MNV_NO_FPU
    #define MNV_PROGMEM_WEIGHTS
    #define MNV_FORCE_BINARY                 /* BNN only at this size */

#elif defined(MNV_TARGET_STM32F0)
    #define MNV_FLASH_BYTES         65536U
    #define MNV_SRAM_BYTES          8192U
    #define MNV_MAX_WEIGHT_BYTES    40960U
    #define MNV_MAX_SRAM_BUDGET     6144U
    #define MNV_ARCH_CORTEX_M0
    #define MNV_NO_FPU

#elif defined(MNV_TARGET_STM32F4)
    #define MNV_FLASH_BYTES         1048576U
    #define MNV_SRAM_BYTES          196608U
    #define MNV_MAX_WEIGHT_BYTES    819200U
    #define MNV_MAX_SRAM_BUDGET     163840U
    #define MNV_ARCH_CORTEX_M4
    /* FPU available but we still use Q8 for security (constant-time) */

#elif defined(MNV_TARGET_HOST)
    #define MNV_FLASH_BYTES         (1024U * 1024U * 16U)
    #define MNV_SRAM_BYTES          (1024U * 1024U * 4U)
    #define MNV_MAX_WEIGHT_BYTES    (1024U * 1024U * 8U)
    #define MNV_MAX_SRAM_BUDGET     (1024U * 1024U * 2U)
    #define MNV_ARCH_HOST
    #include <stdint.h>
    #include <stddef.h>

#else
    #error "No MNV_TARGET defined. Edit mnv_config.h."
#endif

/* =========================================================================
 * ARCHITECTURE SELECTION
 * Uncomment exactly one architecture.
 * ========================================================================= */

/* Default to MLP only if no arch is explicitly selected */
#if !defined(MNV_ARCH_MLP) && !defined(MNV_ARCH_CNN1D) && !defined(MNV_ARCH_BNN)
#define MNV_ARCH_MLP
#endif
/* #define MNV_ARCH_CNN1D          */
/* #define MNV_ARCH_BNN            */   /* Binary Neural Network */

/* Force BNN on ATtiny85 */
#if defined(MNV_FORCE_BINARY) && !defined(MNV_ARCH_BNN)
    #undef  MNV_ARCH_MLP
    #undef  MNV_ARCH_CNN1D
    #define MNV_ARCH_BNN
#endif

/* =========================================================================
 * QUANTIZATION
 * ========================================================================= */

/* #define MNV_QUANT_BINARY   */   /* 1-bit: weights ∈ {-1, +1}        */
/* #define MNV_QUANT_Q4       */   /* 4-bit fixed point                 */
#ifndef MNV_QUANT_Q8
#define MNV_QUANT_Q8           /* 8-bit fixed point (default)       */
#endif
/* #define MNV_QUANT_Q15      */   /* 16-bit, ATmega2560+ only          */

#if defined(MNV_FORCE_BINARY)
    #undef MNV_QUANT_Q8
    #define MNV_QUANT_BINARY
#endif

/* =========================================================================
 * MODEL TOPOLOGY
 * Define your model dimensions here.
 * The Python compiler (minerva_compile.py) auto-generates these.
 * ========================================================================= */

#ifndef MNV_INPUT_SIZE
#define MNV_INPUT_SIZE          16U
#endif
#ifndef MNV_NUM_LAYERS
#define MNV_NUM_LAYERS          3U
#endif
#ifndef MNV_LAYER_0_SIZE
#define MNV_LAYER_0_SIZE        32U
#endif
#ifndef MNV_LAYER_1_SIZE
#define MNV_LAYER_1_SIZE        16U
#endif
#ifndef MNV_OUTPUT_SIZE
#define MNV_OUTPUT_SIZE         4U
#endif

/* CNN1D topology -- set these to match your model */
#ifndef MNV_CNN_KERNEL_SIZE
#define MNV_CNN_KERNEL_SIZE     4U
#endif
#ifndef MNV_CNN_NUM_FILTERS
#define MNV_CNN_NUM_FILTERS     8U
#endif
#ifndef MNV_CNN_POOL_SIZE
#define MNV_CNN_POOL_SIZE       2U
#endif

/* Dense layer right-shift: ceil(log2(flat_size))+7
 * Accounts for accumulated scale in pool->dense transition.
 * Computed by compiler; override with -DMNV_CNN_DENSE_SHIFT=N */
#ifndef MNV_CNN_DENSE_SHIFT
#define MNV_CNN_DENSE_SHIFT     14U
#endif

/* =========================================================================
 * SECURITY FEATURES
 * All enabled by default. Disable only for benchmarking on HOST target.
 * ========================================================================= */

#define MNV_ENABLE_WEIGHT_ENCRYPTION    /* ChaCha20-based weight decryption     */
#define MNV_ENABLE_INTEGRITY_CHECK      /* BLAKE2s model MAC verification        */
#define MNV_ENABLE_CANARIES             /* SRAM canary anti-glitch detection     */
#define MNV_ENABLE_DOUBLE_RUN           /* Redundant inference comparison        */
#define MNV_ENABLE_INPUT_VALIDATION     /* Constant-time input range clamping    */
#define MNV_ENABLE_CONSTANT_TIME        /* Enforce CT arithmetic throughout      */

/* Minimum output confidence to accept result (0-255 in Q8).
 * Default: 0 (disabled). Set this to the MNV_RECOMMENDED_CONFIDENCE value
 * emitted by the compiler in weights.h, or calibrate manually.
 * Setting this too high rejects valid low-magnitude logits from linear
 * output layers — a common issue with Q8 quantized models.           */
#ifndef MNV_MIN_CONFIDENCE
#define MNV_MIN_CONFIDENCE              0U
#endif

/* =========================================================================
 * CRYPTO PARAMETERS
 * ========================================================================= */

#define MNV_CHACHA20_KEY_SIZE   32U   /* 256-bit key */
#define MNV_CHACHA20_IV_SIZE    12U   /* 96-bit nonce */
#define MNV_BLAKE2S_DIGEST_SIZE 32U   /* 256-bit MAC */
#define MNV_CANARY_VALUE        0xDEADBEEFUL
#define MNV_CANARY_COUNT        4U    /* canaries per SRAM region */

/* =========================================================================
 * DERIVED SIZES (do not edit)
 * ========================================================================= */

#define MNV_LAYER_0_WEIGHT_COUNT  (MNV_INPUT_SIZE  * MNV_LAYER_0_SIZE)
#define MNV_LAYER_1_WEIGHT_COUNT  (MNV_LAYER_0_SIZE * MNV_LAYER_1_SIZE)
#define MNV_LAYER_2_WEIGHT_COUNT  (MNV_LAYER_1_SIZE * MNV_OUTPUT_SIZE)
#define MNV_TOTAL_WEIGHT_COUNT    (MNV_LAYER_0_WEIGHT_COUNT + \
                                   MNV_LAYER_1_WEIGHT_COUNT + \
                                   MNV_LAYER_2_WEIGHT_COUNT)
#define MNV_TOTAL_BIAS_COUNT      (MNV_LAYER_0_SIZE + MNV_LAYER_1_SIZE + MNV_OUTPUT_SIZE)

#define MNV_CRYPTO_OVERHEAD       (MNV_CHACHA20_IV_SIZE + MNV_BLAKE2S_DIGEST_SIZE)
#define MNV_NET_WEIGHT_BUDGET     (MNV_MAX_WEIGHT_BYTES - MNV_CRYPTO_OVERHEAD)


/* =========================================================================
 * v1.1 SECURITY FEATURES
 * ========================================================================= */

#define MNV_ENABLE_BLINDED_LUT          /* Masked activation LUT (Law II hardening) */
#define MNV_ENABLE_OUTPUT_AUTH          /* Session MAC over inference output         */

/* Xorshift32 PRNG for LUT blinding — override seed with hardware entropy           */
/* AVR: seed from free-running ADC. STM32: seed from RNG peripheral.                */
#define MNV_PRNG_SEED_DEFAULT   0xDEADC0DEUL

/* Output session MAC — 8-byte truncated BLAKE2s, appended to output buffer         */
#define MNV_OUTPUT_MAC_SIZE     8U

#endif /* MNV_CONFIG_H */
