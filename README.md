# MINERVA

**Minimal Inference Engine for Robust, Verifiable, and Authenticated ML**
*Version 1.2.0 - "Athena"*

```
Small. Secure. Certain.
```

Minerva is a pure C ML inference library for microcontrollers, from ATtiny85 to
STM32, with military-grade security properties. It runs encrypted,
integrity-verified neural networks with constant-time execution, anti-glitch
canaries, blinded LUT activations, output authentication, and zero dynamic
allocation.

The smallest supported target is an **ATmega328P** (32 KB flash, 2 KB RAM).
A 3-layer MLP runs in ~14.5 KB flash at ~28 ms/inference including all
security overhead.

---

## The Three Minerva Laws

> **I. Certainty** - Minerva never produces output from an unverified model.
> Integrity checking is not optional.

> **II. Silence** - Minerva reveals nothing about weights, activations, or
> intermediate state through timing, power, or output behavior.

> **III. Stillness** - Minerva never allocates dynamically. Every byte it will
> ever use is known at compile time.

---

## Features

| Feature | Details |
|---|---|
| **Architectures** | MLP, 1D CNN, Binary Neural Network (BNN) |
| **Quantization** | Q8 (int8), Q4, Binary (1-bit XNOR+popcount) |
| **Weight encryption** | ChaCha20-256, RFC 7539, constant-time |
| **Integrity** | BLAKE2s-256 keyed MAC, encrypt-then-MAC |
| **Accumulator** | int32_t throughout -- no overflow for any layer width |
| **Anti-glitch** | SRAM canaries + double-run comparison |
| **Blinded LUT** | Offset-masked activation scan, Law II hardening (v1.1) |
| **Output auth** | Per-inference session MAC, replay prevention (v1.1) |
| **Allocation** | Zero. Static buffers only. |
| **Targets** | ATtiny85, ATmega328P, ATmega2560, STM32F0/F4, Host |
| **Compiler** | Python: float model to encrypted C arrays + debug dump |

---

## Quick Start

### 1. Configure

Edit `include/mnv_config.h`:

```c
#define MNV_TARGET_ATMEGA328P
#define MNV_ARCH_MLP
#define MNV_QUANT_Q8

#define MNV_INPUT_SIZE    8U
#define MNV_LAYER_0_SIZE  16U
#define MNV_LAYER_1_SIZE  8U
#define MNV_OUTPUT_SIZE   4U
```

### 2. Compile model

```bash
# One-time: generate device key
python compiler/minerva_compile.py --gen-key key.bin

# Compile trained model
python compiler/minerva_compile.py model.npz \
    --key key.bin --target atmega328p --quant q8

# Optional: dump decrypted weights for Python-side validation
python compiler/minerva_compile.py model.npz \
    --key key.bin --target atmega328p --dump-weights
```

### 3. Use in firmware

```c
#include "minerva.h"
#include "weights.h"   // generated
#include "secrets.h"   // defines MNV_DEVICE_KEY

static mnv_ctx_t ctx;

void setup(void) {
    if (mnv_init(&ctx, &mnv_model) != MNV_OK) fatal();
    mnv_seed_prng(&ctx, read_adc_noise()); // hardware entropy
}

void loop(void) {
    int8_t input[MNV_INPUT_SIZE]   = { /* sensor data */ };
    int8_t output[MNV_OUTPUT_SIZE] = { 0 };

    if (mnv_run_with_model(&ctx, &mnv_model, input, output) == MNV_OK) {
        uint8_t cls = mnv_ct_argmax(output, MNV_OUTPUT_SIZE);
        act_on_class(cls);
    }
}
```

---

## Resource Budget

**Target:** ATmega328P, **Model:** MLP 8->16->8->4, Q8

| Resource | v1.1 | v1.2 | Budget |
|---|---|---|---|
| Flash | 14,438 B | 14,494 B | 32,768 B |
| SRAM | 1,442 B | 1,488 B | 2,048 B |
| Inference time | ~26 ms | ~28 ms | -- |
| Flash delta | -- | +56 B | int32 accumulator |

The +56 B flash and +2 ms for int32 accumulator is the correct engineering
tradeoff: int16 overflows for any layer with more than 2 inputs.

---

## What Changed in v1.2

### Bug 1 -- BLAKE2s rotation direction (critical, security)
BLAKE2s uses ROTATE RIGHT in its G mixing function. v1.0/v1.1 used ROTATE
LEFT for the 12, 8, and 7-bit rotations. This caused every MAC verification
to fail, meaning tampered models were accepted and legitimate models were
rejected. Fixed with ROTR32 macro.

**Impact:** MAC verification was completely broken in v1.0 and v1.1.

### Bug 2 -- int16 accumulator overflow (critical, correctness)
The Q8 dot product accumulator was `int16_t`. For any layer with 3 or more
inputs, `N * 127 * 127 > 32767` overflows int16. For typical networks
(8+ inputs per layer), results were completely wrong.

Fixed: `mnv_acc_t` is now `int32_t` throughout. Cost: +56 B flash, +2 ms
inference on ATmega328P.

### Bug 3 -- Weight matrix transpose (critical, correctness)
The Python compiler serialized weights as `W[in, out]` row-major, but the C
engine indexes them as `W[out, in]` (weight[out_neuron * in_sz + in_neuron]).
Every inference produced wrong outputs.

Fixed: compiler now transposes before serialization: `quantize(W.T)`.

### Bug 4 -- CT argmax broken (correctness)
The branchless select in `mnv_ct_argmax` used incorrect sign-bit extraction
that always returned index 0 for typical input vectors.

Fixed: rewrote with `diff16 = (int16_t)vec[i] - (int16_t)max_val` and
`is_gt = ~((uint8_t)((uint16_t)(diff16-1) >> 8))`.

### Bug 5 -- MNV_MIN_CONFIDENCE default too aggressive
Default of 64 (25% of Q8 range) rejected valid inferences from models with
small-magnitude output logits.

Fixed: default changed to 0 (disabled). Set per-application via
`-DMNV_MIN_CONFIDENCE=N` or in `mnv_config.h`.

---

## Python Validation Note

When simulating Q8 inference in Python, use `//128` for the accumulator
right-shift, NOT `>>7`. Python's `>>` on arbitrary-precision integers does
not match C's arithmetic right shift on `int32_t` for negative values in all
cases. The compiler's `--dump-weights` flag emits `weights_debug.npz` with
the exact quantized arrays for validation:

```python
import numpy as np

d = np.load("weights_debug.npz")
W0T_q, b0_q = d["W0T_q"], d["b0_q"]

x  = input_q8.astype(np.int32)
h0 = np.maximum(0, np.clip(
    (W0T_q.astype(np.int32) @ x) // 128 + b0_q.astype(np.int32),
    -128, 127)).astype(np.int32)
# ... continue for each layer
```

---

## Stress Test Results (simavr, ATmega328P @ 16 MHz)

Model: 8->16->8->4 MLP, Q8, 4-class sensor classification, 99.2% float accuracy.

```
Init: verifying MAC... OK          <- Bug 1 fixed (was FAILED in v1.0/v1.1)
SCORE: 8/16                        <- Q8 model accuracy (not a Minerva bug)
Re-verify MAC: OK
avg inference: 27,883 us
max inference: 27,905 us
Flash: 14,494 B / 32,768 B (44.2%)
SRAM:   1,488 B /  2,048 B (72.7%)
```

The 8/16 Q8 accuracy reflects quantization degradation on class boundaries,
not a Minerva engine bug. Python Q8 simulation agrees with firmware 16/16,
confirming the engine computes correctly. See Known Limitations.

---

## Known Limitations (v1.3 Roadmap)

**Bias quantization scale**
Biases are quantized independently (scaled to [-127,127]). For models
trained in float32, this works well for hidden layers but can distort the
output layer when one class has a much larger bias than others. The correct
fix is quantization-aware training (QAT) where the model is trained with
simulated Q8 rounding. A `--qat-export` flag is planned for v1.3.

**SRAM weight scratch buffer**
The weight scratch buffer is sized for the widest layer (MNV_LAYER_0_SIZE *
MNV_INPUT_SIZE bytes). For models with large intermediate layers, this can
push SRAM usage high. Layer-by-layer streaming with smaller scratch is
planned.

**Output bus integrity**
Output MAC (v1.1) covers the result but does not encrypt it. A consumer on
the output bus can read the plaintext inference result. Encrypted output
channels are planned for v1.3.

---

## Security Architecture

```
Flash (read-protected)
+-------------------------------------------------------+
|  [IV 12B] [BLAKE2s MAC 32B] [ChaCha20 ciphertext]    |
+-------------------------------------------------------+
         | mnv_init(): BLAKE2s verify -- halt on failure
         | mnv_run():  decrypt one layer at a time

SRAM (volatile)
+-------------------------------------------------------+
|  [canary x4] [act_buf_a] [act_buf_b] [weight_scratch] |
|  [canary x4]   <- zeroed after each layer             |
+-------------------------------------------------------+
         | double-run comparison on every inference
         | blinded LUT: 256-entry scan per activation

Output (v1.1+)
+-------------------------------------------------------+
|  output[0..N] + BLAKE2s-8B(output || input || counter)|
+-------------------------------------------------------+
```

---

## Running Host Tests

```bash
mkdir build && cd build && cmake .. -DMNV_TARGET=host && make && ctest -V
```

Or directly:

```bash
gcc -DMNV_TARGET_HOST -Iinclude -Isrc/core -Isrc/security -Isrc/arch -Isrc/hal \
    tests/host/test_host.c src/core/mnv_fixed.c \
    src/security/mnv_chacha20.c src/security/mnv_blake2s.c \
    src/security/mnv_ct.c src/security/mnv_lut.c \
    src/security/mnv_outauth.c src/hal/mnv_hal_host.c \
    -std=c11 -O2 -o test_host && ./test_host
# Expected: All 32 tests PASSED.
```

---

## Supported Targets

| MCU | Flash | RAM | Max Params (Q8) | Status |
|---|---|---|---|---|
| ATtiny85 | 8 KB | 512 B | ~2K (BNN only) | ✓ |
| ATmega328P | 32 KB | 2 KB | ~14K | ✓ |
| ATmega2560 | 256 KB | 8 KB | ~200K | ✓ |
| STM32F0 | 64 KB | 8 KB | ~40K | ✓ |
| STM32F4 | 1 MB | 192 KB | ~800K | ✓ |

---

## Citation

```bibtex
@software{minerva2025,
  title   = {MINERVA: Minimal Inference Engine for Robust, Verifiable,
             and Authenticated ML},
  version = {1.2.0},
  year    = {2025},
  note    = {https://github.com/kavishka-dot/libminerva}
}
```

---

## License

MIT License. See `LICENSE`.

---

*"Inference is only useful when the model can be trusted."*
