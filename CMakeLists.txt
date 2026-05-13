cmake_minimum_required(VERSION 3.16)
project(Minerva C)

set(CMAKE_C_STANDARD 11)

# ── Host build (testing) ────────────────────────────────────────────────────
if(NOT DEFINED MNV_TARGET)
    set(MNV_TARGET "host")
endif()

message(STATUS "Minerva target: ${MNV_TARGET}")

# Common source files
set(MNV_SOURCES
    src/core/mnv_fixed.c
    src/core/mnv_engine.c
    src/arch/mnv_mlp.c
    src/arch/mnv_cnn1d.c
    src/arch/mnv_bnn.c
    src/security/mnv_chacha20.c
    src/security/mnv_blake2s.c
    src/security/mnv_ct.c
)

# Host-specific sources
if(MNV_TARGET STREQUAL "host")
    list(APPEND MNV_SOURCES src/hal/mnv_hal_host.c)

    add_library(minerva STATIC ${MNV_SOURCES})
    target_include_directories(minerva PUBLIC include src)
    target_compile_definitions(minerva PUBLIC MNV_TARGET_HOST MNV_ARCH_HOST)
    target_compile_options(minerva PRIVATE -Wall -Wextra -O2 -fsanitize=address,undefined)
    target_link_options(minerva PUBLIC -fsanitize=address,undefined)

    # Test executable
    add_executable(minerva_test tests/host/test_host.c)
    target_link_libraries(minerva_test minerva)
    target_compile_definitions(minerva_test PRIVATE MNV_TARGET_HOST)

    enable_testing()
    add_test(NAME minerva_host_tests COMMAND minerva_test)

    message(STATUS "Host build configured. Run: cmake --build build && ctest --test-dir build")
endif()

# AVR cross-compilation is handled via examples/*/Makefile and avr.mk
# See: make -f avr.mk MCU=atmega328p
