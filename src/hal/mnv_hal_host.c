/**
 * @file mnv_hal_host.c
 * @brief Host (Linux/macOS/Windows) HAL — for unit testing only
 */

#include "mnv_hal.h"

#if defined(MNV_TARGET_HOST)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t mnv_hal_flash_read_byte(const uint8_t *ptr) { return *ptr; }

void mnv_hal_flash_read_block(const uint8_t *src, uint8_t *dst, uint16_t len)
{
    memcpy(dst, src, len);
}

void mnv_hal_fatal(void)
{
    fprintf(stderr, "[MINERVA] FATAL: security violation — halting.\n");
    abort();
}

void mnv_hal_critical_enter(void) { /* no-op on host */ }
void mnv_hal_critical_exit(void)  { /* no-op on host */ }

#endif
