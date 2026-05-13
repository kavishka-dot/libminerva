# Minerva Porting Guide

## Adding a New Target MCU

1. **Add target block in `mnv_config.h`**:
   ```c
   #elif defined(MNV_TARGET_MY_MCU)
       #define MNV_FLASH_BYTES      65536U
       #define MNV_SRAM_BYTES       4096U
       #define MNV_MAX_WEIGHT_BYTES 40000U
       #define MNV_MAX_SRAM_BUDGET  3000U
       #define MNV_ARCH_CORTEX_M0
       #define MNV_NO_FPU
   ```

2. **Implement HAL** in `src/hal/mnv_hal_<target>.c`:
   - `mnv_hal_flash_read_byte()` — reads from flash address
   - `mnv_hal_flash_read_block()` — block read
   - `mnv_hal_fatal()` — irrecoverable halt
   - `mnv_hal_critical_enter/exit()` — disable/enable interrupts

3. **Add to build system** (CMakeLists.txt or Makefile).

4. **Run host tests** first, then cross-compile and run on target.

## Key Provisioning

### ATmega (EEPROM)
```c
// Write key to EEPROM during factory programming
eeprom_write_block(key_bytes, (void*)0x00, 32);
// Then set EEPROM write-lock fuse bits
```

### STM32 (Flash OTP)
```c
// Write to OTP area (one-time programmable)
HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, OTP_BASE, key_word);
// Enable RDP Level 2
```

### Secure Element (ATECC608)
The key never leaves the secure element. Use ATECC608 HMAC or ECDH to derive a session key for each boot, then pass the derived key to `mnv_init()`.

## Reducing Flash Usage

- Use `MNV_QUANT_BINARY` — saves ~14× weight space
- Disable unused arch backends in Makefile (`-DMNV_ARCH_MLP` only)
- Use `--gc-sections` linker flag (already in avr.mk)
- Profile with `avr-size --format=avr` and prune unused modules
