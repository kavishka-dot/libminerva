/**
 * @file test_host.c
 * @brief Minerva host unit tests — v1.1
 *
 * Build:
 *   gcc -DMNV_TARGET_HOST -I../../include -I../../src \
 *       test_host.c \
 *       ../../src/core/mnv_fixed.c \
 *       ../../src/security/mnv_chacha20.c \
 *       ../../src/security/mnv_blake2s.c \
 *       ../../src/security/mnv_ct.c \
 *       ../../src/security/mnv_lut.c \
 *       ../../src/security/mnv_outauth.c \
 *       ../../src/hal/mnv_hal_host.c \
 *       -o test_host && ./test_host
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define MNV_TARGET_HOST
#include "../../include/mnv_config.h"
#include "../../include/mnv_types.h"
#include "../../src/core/mnv_fixed.h"
#include "../../src/security/mnv_chacha20.h"
#include "../../src/security/mnv_blake2s.h"
#include "../../src/security/mnv_ct.h"
#include "../../src/security/mnv_lut.h"
#include "../../src/security/mnv_prng.h"
#include "../../src/security/mnv_outauth.h"

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"
#define TEST(n) printf("  %-52s ", n)
#define ASSERT_EQ(a,b)  do { if((a)==(b)) printf(PASS"\n"); else { printf(FAIL" (got %d want %d)\n",(int)(a),(int)(b)); failures++; } } while(0)
#define ASSERT_OK(s)    ASSERT_EQ((s), MNV_OK)
#define ASSERT_NEQ(a,b) do { if((a)!=(b)) printf(PASS"\n"); else { printf(FAIL" (expected !=)\n"); failures++; } } while(0)

static int failures = 0;

static void test_chacha20(void)
{
    printf("\n[ChaCha20 — RFC 7539]\n");
    static const uint8_t key[32] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
    };
    static const uint8_t nonce[12] = { 0,0,0,0,0,0,0,0x4a,0,0,0,0 };
    static const uint8_t want[4]   = { 0x22,0x4f,0x51,0xf3 };
    uint8_t block[64];
    mnv_chacha20_block(key, nonce, 1, block);
    TEST("RFC 7539 block[0..3]");
    printf("%s\n", memcmp(block, want, 4)==0 ? PASS : FAIL);
    if (memcmp(block, want, 4)!=0) failures++;

    TEST("Encrypt/decrypt round trip 128B");
    uint8_t plain[128], cipher[128], dec[128];
    for (int i=0;i<128;i++) plain[i]=(uint8_t)i;
    mnv_chacha20_ctx_t e,d;
    mnv_chacha20_init(&e,key,nonce,0); mnv_chacha20_init(&d,key,nonce,0);
    mnv_chacha20_decrypt(&e,plain,cipher,128);
    mnv_chacha20_decrypt(&d,cipher,dec,128);
    printf("%s\n", memcmp(plain,dec,128)==0 ? PASS : FAIL);
    if (memcmp(plain,dec,128)!=0) failures++;

    TEST("Different counters → different keystreams");
    uint8_t b0[64],b1[64];
    mnv_chacha20_block(key,nonce,0,b0); mnv_chacha20_block(key,nonce,1,b1);
    ASSERT_NEQ(b0[0],b1[0]);
}

static void test_blake2s(void)
{
    printf("\n[BLAKE2s-256]\n");
    static const uint8_t want_empty[32] = {
        0x69,0x21,0x7a,0x30,0x79,0x90,0x80,0x94,0xe1,0x11,0x21,0xd0,0x42,0x35,0x4a,0x7c,
        0x1f,0x55,0xb6,0x48,0x2c,0xa1,0xa5,0x1e,0x1b,0x25,0x0d,0xfd,0x1e,0xd0,0xee,0xf9
    };
    uint8_t digest[32];
    mnv_blake2s_ctx_t ctx;
    mnv_blake2s_init(&ctx,NULL,0); mnv_blake2s_final(&ctx,digest);
    TEST("BLAKE2s(\"\") known-answer");
    printf("%s\n", memcmp(digest,want_empty,32)==0 ? PASS : FAIL);
    if (memcmp(digest,want_empty,32)!=0) failures++;

    static const uint8_t key[32]={0x01};
    static const uint8_t data[]="MINERVA Athena";
    uint8_t mac[32];
    mnv_blake2s_mac(key,32,data,sizeof(data)-1,mac);
    TEST("MAC: correct → OK");      ASSERT_OK(mnv_blake2s_verify(key,32,data,sizeof(data)-1,mac));
    TEST("MAC: tampered → TAMPER"); uint8_t bad[]="MINERVA Athena!";
    ASSERT_EQ(mnv_blake2s_verify(key,32,bad,sizeof(bad)-1,mac), MNV_ERR_TAMPER);
    TEST("MAC: wrong key → TAMPER"); uint8_t wk[32]={0x02};
    ASSERT_EQ(mnv_blake2s_verify(wk,32,data,sizeof(data)-1,mac), MNV_ERR_TAMPER);
}

static void test_fixed(void)
{
    printf("\n[Fixed-Point Q8]\n");
    TEST("Clamp  200 → 127");   ASSERT_EQ(mnv_q8_clamp(200), 127);
    TEST("Clamp -200 → -128");  ASSERT_EQ(mnv_q8_clamp(-200), -128);
    TEST("ReLU -50 → 0");       ASSERT_EQ(mnv_act_relu(-50), 0);
    TEST("ReLU  60 → 60");      ASSERT_EQ(mnv_act_relu(60), 60);
    TEST("Sign   0 → 127");     ASSERT_EQ(mnv_act_sign(0), 127);
    TEST("Sign  -1 → -128");    ASSERT_EQ(mnv_act_sign(-1), -128);
    TEST("Sigmoid(0) ≈ 0");
    int8_t s=mnv_act_sigmoid(0); printf("%s (got %d)\n", (s>=-10&&s<=10)?PASS:FAIL,s);
    if(!(s>=-10&&s<=10)) failures++;
    TEST("Q8 dot [1,2,3]·[1,2,3]=14");
    static const mnv_weight_t w[]={1,2,3}; static const mnv_act_t a[]={1,2,3};
    ASSERT_EQ(mnv_q8_dot(w,a,3), 14);
}

static void test_ct(void)
{
    printf("\n[Constant-Time Primitives]\n");
    uint8_t a[]={1,2,3,4},b[]={1,2,3,4},c[]={1,2,3,5};
    TEST("CT compare equal → 0");      ASSERT_EQ(mnv_ct_compare(a,b,4), 0);
    TEST("CT compare differ → non-0"); ASSERT_NEQ(mnv_ct_compare(a,c,4), 0);
    TEST("CT argmax [10,50,20,5] → 1");
    mnv_act_t v[]={10,50,20,5}; ASSERT_EQ(mnv_ct_argmax(v,4), 1);
    TEST("Input validation valid → OK");
    mnv_act_t iv[]={0,64,-64,127,-128}; ASSERT_OK(mnv_ct_validate_input(iv,5));
}

static void test_prng(void)
{
    printf("\n[v1.1 Xorshift32 PRNG]\n");
    uint32_t st=0xDEADBEEF;
    TEST("Non-zero seed → non-zero output"); uint32_t r=mnv_prng_next(&st); ASSERT_NEQ((int)r,0);
    TEST("State advances each call");
    uint32_t s2=0xDEADBEEF; uint32_t r1=mnv_prng_next(&s2),r2=mnv_prng_next(&s2);
    ASSERT_NEQ((int)r1,(int)r2);
    TEST("Same seed → same sequence");
    uint32_t sa=0x12345678,sb=0x12345678; int ok=1;
    for(int i=0;i<20;i++) if(mnv_prng_next(&sa)!=mnv_prng_next(&sb)){ok=0;break;}
    printf("%s\n", ok?PASS:FAIL); if(!ok) failures++;
}

static void test_blinded_lut(void)
{
    printf("\n[v1.1 Blinded LUT — Law II Hardening]\n");
    uint32_t p=0xCAFEBABE;

    TEST("Blinded sigmoid(0) == naive");
    ASSERT_EQ(mnv_act_sigmoid(0), mnv_lut_sigmoid_blinded(0,&p));
    TEST("Blinded sigmoid(64) == naive");
    ASSERT_EQ(mnv_act_sigmoid(64), mnv_lut_sigmoid_blinded(64,&p));
    TEST("Blinded tanh(0) == naive");
    ASSERT_EQ(mnv_act_tanh(0), mnv_lut_tanh_blinded(0,&p));
    TEST("Blinded sigmoid: correct all 256 inputs");
    int ok=1;
    for(int x=-128;x<=127;x++){uint32_t ps=(uint32_t)(x+300);
        if(mnv_act_sigmoid((int8_t)x)!=mnv_lut_sigmoid_blinded((int8_t)x,&ps)){ok=0;break;}}
    printf("%s\n",ok?PASS:FAIL); if(!ok) failures++;
    TEST("Blinded tanh: correct all 256 inputs");
    ok=1;
    for(int x=-128;x<=127;x++){uint32_t ps=(uint32_t)(x+400);
        if(mnv_act_tanh((int8_t)x)!=mnv_lut_tanh_blinded((int8_t)x,&ps)){ok=0;break;}}
    printf("%s\n",ok?PASS:FAIL); if(!ok) failures++;
}

static void test_outauth(void)
{
    printf("\n[v1.1 Output Authentication]\n");
    static mnv_ctx_t ctx;
    memset(&ctx,0,sizeof(ctx));
    ctx.initialized=true; ctx.verified=true; ctx.inference_counter=0;
    mnv_canary_plant(&ctx);

    static const uint8_t dk[32]={0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
                                   0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10,
                                   0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
                                   0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20};
    mnv_act_t in[MNV_INPUT_SIZE]={10,20,30,40};
    mnv_act_t out[MNV_OUTPUT_SIZE]={100,-50,20,-10};

    mnv_outauth_compute(&ctx,dk,out,in);

    TEST("MAC non-zero after compute");
    int nz=0; for(int i=0;i<MNV_OUTPUT_MAC_SIZE;i++) if(ctx.output_mac[i]) nz=1;
    printf("%s\n",nz?PASS:FAIL); if(!nz) failures++;

    TEST("Verify correct data → OK");   ASSERT_OK(mnv_outauth_verify(&ctx,dk,out,in));
    TEST("Tampered output → TAMPER");
    mnv_act_t bo[MNV_OUTPUT_SIZE]={99,-50,20,-10};
    ASSERT_EQ(mnv_outauth_verify(&ctx,dk,bo,in), MNV_ERR_TAMPER);
    TEST("Tampered input → TAMPER");
    mnv_act_t bi[MNV_INPUT_SIZE]={11,20,30,40};
    ASSERT_EQ(mnv_outauth_verify(&ctx,dk,out,bi), MNV_ERR_TAMPER);
    TEST("Counter increments each call");
    mnv_outauth_compute(&ctx,dk,out,in);
    ASSERT_EQ((int)ctx.inference_counter, 2);
}

int main(void)
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  MINERVA v1.3.0-Athena — Host Test Suite              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    test_chacha20();
    test_blake2s();
    test_fixed();
    test_ct();
    test_prng();
    test_blinded_lut();
    test_outauth();
    printf("\n────────────────────────────────────────────────────────────\n");
    if (failures==0) printf("  All tests \033[32mPASSED\033[0m. Minerva v1.3 is ready.\n");
    else             printf("  \033[31m%d test(s) FAILED.\033[0m\n", failures);
    printf("────────────────────────────────────────────────────────────\n");
    return failures;
}
