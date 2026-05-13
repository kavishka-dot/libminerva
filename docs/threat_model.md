# Minerva Threat Model

**Version:** 1.0.0-McGonagall  
**Status:** Normative

---

## 1. Scope

This document defines the threat model for Minerva ML inference on microcontrollers. It describes the assets being protected, the adversary model, the attack surface, and the mitigations implemented in v1.0.

---

## 2. Assets

| Asset | Confidentiality | Integrity | Availability |
|---|---|---|---|
| Model weights | Critical | Critical | High |
| Inference result | — | Critical | High |
| Device key | Critical | Critical | — |
| Input sensor data | Context-dependent | High | — |
| Intermediate activations | High | High | — |

**Model weights** are the primary IP asset. Disclosure allows model extraction and replication. Modification allows the adversary to control inference behavior.

**Inference result** integrity is critical in autonomous decision contexts (access control, anomaly detection, actuation).

**Device key** compromise breaks all cryptographic protections.

---

## 3. Adversary Model

### 3.1 Adversary Capabilities

Minerva assumes an adversary with:

- **Physical access** to the device (can connect probes, logic analyzers)
- **Flash dump capability** (ISP, JTAG, or exploit of bootloader)
- **Power and EM measurement** equipment (oscilloscope, EM probe)
- **Fault injection tools** (voltage glitcher, EM fault injector, laser)
- **Software access** to the communication interface (UART, SPI, I²C)
- **Time** (offline analysis of captured traces)

### 3.2 Adversary Goals

1. Extract model weights from flash
2. Determine model architecture and parameters
3. Craft adversarial inputs that produce desired outputs
4. Modify model to produce attacker-controlled outputs
5. Bypass integrity checks to run a tampered model
6. Recover intermediate activations via side channels

### 3.3 Out of Scope (v1.0)

- Compromise of the key provisioning process (assumed secure)
- Supply chain attacks on the MCU itself
- Attacks requiring access to the firmware compilation environment
- Multi-device key extraction (side-channel across many identical devices)
- Attacks on the training environment or training data

---

## 4. Attack Surface

### 4.1 Flash Dump → Weight Extraction

**Attack:** Adversary dumps flash via ISP or JTAG and reads weight values.

**Mitigation:** Weights are encrypted with ChaCha20-256. The key is never stored in flash — it is provisioned into a protected memory region (EEPROM with write-lock fuse on AVR, or TrustZone secure world on ARM). Without the key, the weight blob is indistinguishable from random bytes.

**Residual risk:** Key extraction via side channel during decryption (see §4.3).

---

### 4.2 Model Tampering

**Attack:** Adversary modifies the encrypted weight blob in flash to alter inference behavior.

**Mitigation:** BLAKE2s-256 keyed MAC is computed over the entire encrypted weight blob during `mnv_init()` and on every explicit `mnv_verify()` call. Any modification to even one byte causes the MAC check to fail. On failure, the device zeroes its SRAM state and returns `MNV_ERR_TAMPER` — no inference output is produced.

**Note:** MAC is computed over *ciphertext* (encrypt-then-MAC scheme). This prevents chosen-ciphertext attacks against the decryption layer.

---

### 4.3 Side-Channel Attacks (Power/EM)

**Attack:** Adversary measures power consumption or EM emanation during inference and correlates with intermediate values to recover key material or weights.

**Mitigations:**

1. **ChaCha20** uses only ADD, XOR, ROTATE — no data-dependent table lookups. On AVR (no cache), this eliminates cache-timing attacks. Power leakage of arithmetic operations is significantly lower than S-box lookups (cf. AES).

2. **Constant-time arithmetic** throughout the inference pipeline: no data-dependent branches, no early exits, no data-dependent memory access patterns.

3. **LUT-based activations** (sigmoid, tanh) are accessed sequentially by index — the index depends on the activation value, which does leak through power. **This is a known limitation of v1.0.** Mitigation planned for v1.1: input-blinded LUT access.

4. **Weight decryption per layer**: only one layer's weights are ever in SRAM simultaneously. The decryption scratch is zeroed immediately after the forward pass through each layer, limiting the window for power analysis.

**Residual risk (v1.0):** DPA against activation LUT accesses is theoretically possible with many traces. The attack complexity is high but not infeasible for a well-resourced adversary. See v1.1 roadmap.

---

### 4.4 Fault Injection (Voltage/Clock Glitching)

**Attack:** Adversary injects a fault during the MAC verification check, causing the integrity check to report success even when it fails, or skips the check entirely.

**Mitigations:**

1. **SRAM canaries**: `uint32_t` sentinel values are planted at known locations before inference and checked after every layer. A fault that corrupts SRAM (common with voltage glitching) will corrupt canaries and trigger `MNV_ERR_GLITCH`.

2. **Double-run comparison**: inference is executed twice with independent ChaCha20 counter streams. Results are compared using constant-time compare. A single-event fault that alters the computation of one run is detected.

3. **Constant-time MAC comparison**: `mnv_ct_compare()` uses bitwise OR accumulation — a fault that skips one comparison byte cannot cause a false positive.

4. **HAL fatal**: on `MNV_ERR_GLITCH`, `mnv_hal_fatal()` enables the hardware watchdog with 15ms timeout and loops until reset. The device reboots rather than continuing in an undefined state.

**Residual risk:** A precisely timed fault that corrupts the counter value in `mnv_ct_compare()` before the final branch could theoretically bypass the check. This is extremely difficult in practice but not impossible. Hardware secure elements (ATECC608) eliminate this residual risk.

---

### 4.5 Adversarial Inputs

**Attack:** Adversary crafts inputs that cause the model to produce an incorrect output (misclassification).

**Mitigations:**

1. **Input range validation** (constant-time): all input values are checked against `[MNV_Q_MIN, MNV_Q_MAX]`. Out-of-range inputs are rejected with `MNV_ERR_INPUT` before any inference computation begins.

2. **Confidence threshold**: output is rejected if the maximum logit is below `MNV_MIN_CONFIDENCE` (default: 25%). Low-confidence outputs — which adversarial inputs often produce after Q8 quantization — are rejected with `MNV_ERR_CONFIDENCE`.

**Residual risk:** Adversarial examples that remain within the valid input range and produce high-confidence wrong outputs are not detected. This is an inherent limitation of inference-time defenses. Training-time adversarial training is recommended for high-assurance deployments.

---

### 4.6 Communication Channel Attacks

**Attack:** Adversary injects or replays inference results on the output bus.

**Mitigation (planned v1.1):** Output signing with a per-device session key. In v1.0, output integrity is the responsibility of the application layer.

---

## 5. Security Properties (v1.0 Guarantees)

| Property | Guaranteed | Notes |
|---|---|---|
| Weight confidentiality | ✓ | ChaCha20-256 encryption |
| Weight integrity | ✓ | BLAKE2s-256 MAC |
| Tamper detection (flash) | ✓ | MAC check at init and on demand |
| Fault injection detection | ✓ | Canaries + double-run |
| Constant-time arithmetic | ✓ | No data-dependent branches |
| Cache-timing resistance | ✓ | ChaCha20 (no S-box) |
| Power-analysis resistance | Partial | LUT access is v1.1 |
| Adversarial input detection | Partial | Range + confidence check only |
| Output authentication | ✗ | Planned v1.1 |

---

## 6. Key Management Recommendations

The device key (`MNV_DEVICE_KEY`) is the root of all security. Its compromise breaks all protections.

**ATmega:** Use the AVR EEPROM with lock bits set (fuse `BOOTRST=0`, `BLB1=0`). Disable JTAG fuse. Use a unique per-device key derived from a factory master key via HKDF.

**STM32:** Use the RDP (Read-Out Protection) Level 2 option byte. Store key in backup SRAM powered by VBAT.

**Production:** Use a hardware secure element (ATECC608A, SE050) as a key store and co-processor for MAC verification. The MCU never sees the key in cleartext.

---

## 7. Compliance Notes

Minerva's security design is informed by:

- NIST SP 800-193 (Platform Firmware Resiliency Guidelines)
- ETSI EN 303 645 (Cyber Security for Consumer IoT)
- IEC 62443-4-2 (Security for Industrial Automation — Component Level)
- CHES best practices for embedded cryptographic implementations
