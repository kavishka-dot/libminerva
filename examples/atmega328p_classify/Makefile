# Makefile — Minerva ATmega328P example
# Requires: avr-gcc, avrdude

MCU        = atmega328p
F_CPU      = 16000000UL
PROGRAMMER = arduino
PORT       = /dev/ttyUSB0
BAUD       = 115200

CC         = avr-gcc
OBJCOPY    = avr-objcopy
SIZE       = avr-size

MINERVA    = ../..
CFLAGS     = -mmcu=$(MCU) -DF_CPU=$(F_CPU) \
             -DMNV_TARGET_ATMEGA328P \
             -Os -std=c11 -Wall -Wextra \
             -ffunction-sections -fdata-sections \
             -I$(MINERVA)/include \
             -I$(MINERVA)/src

LDFLAGS    = -mmcu=$(MCU) -Wl,--gc-sections

SRCS = main.c \
       $(MINERVA)/src/core/mnv_fixed.c \
       $(MINERVA)/src/core/mnv_engine.c \
       $(MINERVA)/src/arch/mnv_mlp.c \
       $(MINERVA)/src/security/mnv_chacha20.c \
       $(MINERVA)/src/security/mnv_blake2s.c \
       $(MINERVA)/src/security/mnv_ct.c \
       $(MINERVA)/src/hal/mnv_hal_avr.c \
       weights.c

OBJS = $(SRCS:.c=.o)

all: minerva_classify.hex
	@$(SIZE) --mcu=$(MCU) --format=avr minerva_classify.elf

minerva_classify.elf: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

minerva_classify.hex: minerva_classify.elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

flash: minerva_classify.hex
	avrdude -p $(MCU) -c $(PROGRAMMER) -P $(PORT) -b $(BAUD) \
	        -U flash:w:$<:i

# Generate weights from a trained model
weights: model.npz key.bin
	python $(MINERVA)/compiler/minerva_compile.py model.npz \
	       --key key.bin --target atmega328p --quant q8 --out-dir .

# Generate a fresh device key
genkey:
	python $(MINERVA)/compiler/minerva_compile.py --gen-key key.bin

clean:
	rm -f $(OBJS) minerva_classify.elf minerva_classify.hex

.PHONY: all flash weights genkey clean
