/**
 * @file mnv_hal.h
 * @brief Hardware abstraction layer interface
 */
#ifndef MNV_HAL_H
#define MNV_HAL_H

#include "mnv_types.h"

uint8_t mnv_hal_flash_read_byte(const uint8_t *flash_ptr);
void    mnv_hal_flash_read_block(const uint8_t *flash_src,
                                  uint8_t *sram_dst, uint16_t len);
void    mnv_hal_fatal(void);
void    mnv_hal_critical_enter(void);
void    mnv_hal_critical_exit(void);

#endif /* MNV_HAL_H */
