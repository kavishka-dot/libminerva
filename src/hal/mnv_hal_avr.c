/**
 * @file mnv_hal_avr.c
 * @brief AVR (ATmega/ATtiny) hardware abstraction layer
 *
 * Handles PROGMEM flash reads for weight access.
 * On AVR, weights live in flash and must be read byte-by-byte
 * via pgm_read_byte() — direct pointer dereference reads SRAM, not flash.
 */

#include "mnv_hal.h"

#if defined(MNV_ARCH_AVR8)
#include <avr/pgmspace.h>
#include <avr/wdt.h>

/**
 * @brief Read one byte from flash (PROGMEM).
 */
uint8_t mnv_hal_flash_read_byte(const uint8_t *flash_ptr)
{
    return pgm_read_byte(flash_ptr);
}

/**
 * @brief Read a block from flash into SRAM.
 *
 * Used to copy encrypted weight chunks into the decryption scratch buffer.
 */
void mnv_hal_flash_read_block(const uint8_t *flash_src,
                               uint8_t       *sram_dst,
                               uint16_t       len)
{
    memcpy_P(sram_dst, flash_src, len);
}

/**
 * @brief Trigger hardware watchdog reset — used on fatal security error.
 *
 * Enables WDT with minimum timeout (15ms) then loops until reset occurs.
 * This ensures the device halts on tamper detection rather than
 * continuing in an undefined state.
 */
void mnv_hal_fatal(void)
{
    wdt_enable(WDTO_15MS);
    while (1) { /* wait for watchdog */ }
}

/**
 * @brief Disable interrupts — called during sensitive operations.
 */
void mnv_hal_critical_enter(void)
{
    __asm__ __volatile__ ("cli" ::: "memory");
}

/**
 * @brief Re-enable interrupts.
 */
void mnv_hal_critical_exit(void)
{
    __asm__ __volatile__ ("sei" ::: "memory");
}

#endif /* MNV_ARCH_AVR8 */
