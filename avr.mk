# avr.mk — Direct AVR-GCC build for Minerva
# Usage: make -f avr.mk MCU=atmega328p APP=examples/atmega328p_classify

MCU        ?= atmega328p
F_CPU      ?= 16000000UL
APP        ?= examples/atmega328p_classify
PROGRAMMER ?= arduino
PORT       ?= /dev/ttyUSB0
BAUD       ?= 115200

CC      = avr-gcc
OBJCOPY = avr-objcopy
SIZE    = avr-size

DEFS = -DMNV_TARGET_ATMEGA328P -DF_CPU=$(F_CPU)

CFLAGS = -mmcu=$(MCU) $(DEFS) \
         -Os -std=c11 -Wall -Wextra \
         -ffunction-sections -fdata-sections \
         -Iinclude -Isrc

SRCS = src/core/mnv_fixed.c \
       src/core/mnv_engine.c \
       src/arch/mnv_mlp.c \
       src/arch/mnv_cnn1d.c \
       src/arch/mnv_bnn.c \
       src/security/mnv_chacha20.c \
       src/security/mnv_blake2s.c \
       src/security/mnv_ct.c \
       src/hal/mnv_hal_avr.c \
       $(APP)/main.c \
       $(APP)/weights.c

OBJS = $(SRCS:.c=.o)

TARGET = minerva_$(MCU)

all: $(TARGET).hex
	@$(SIZE) --mcu=$(MCU) --format=avr $(TARGET).elf

$(TARGET).elf: $(OBJS)
	$(CC) -mmcu=$(MCU) -Wl,--gc-sections -o $@ $^

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

flash: $(TARGET).hex
	avrdude -p $(MCU) -c $(PROGRAMMER) -P $(PORT) -b $(BAUD) \
	        -U flash:w:$<:i

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).hex

.PHONY: all flash clean
