/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Michael Lamothe
 *
 * The mock host: a C consumer of vr/native-host.h ONLY (compiled as C,
 * linked against the C++ implementation) — the stable desktop boundary's
 * contract exactly as a host renderer sees it: the label, the create/
 * evaluate/destroy lifecycle, the counter schedule with DLSSNR.Reset, the
 * determinism of the synthetic pipeline, the arena report, and the
 * rejection matrix with the output buffer untouched on failure.
 *
 * Built by vr/test-native-host.sh at -O2 (the full mock) and at -O0 (the
 * recorded codegen level; its run was reinstated once the uninitialised-read
 * investigation was pinned, NATIVE-LIVE.md). The synthetic weights make the
 * transformation arbitrary; the label must say so.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "native-host.h"

static int g_failures = 0;
static long long g_checks = 0;

static void check(int ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        printf("FAILED (%d): %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check((expr) != 0, #expr, __LINE__)

static void check_status(dlss5_nr_status got, dlss5_nr_status want,
                         const char* expr, int line) {
    ++g_checks;
    if (got != want) {
        printf("FAILED (%d): %s (got %d, want %d)\n", line, expr, (int)got,
               (int)want);
        ++g_failures;
    }
}
#define CHECK_STATUS(got, want) \
    check_status((got), (want), #got, __LINE__)

enum { kWidth = 64, kHeight = 72, kBytes = 3 * kWidth * kHeight };

static void fill_frame(unsigned char* rgb, int frame, int salt) {
    for (int i = 0; i < kBytes; ++i) {
        rgb[i] = (unsigned char)((i * 13 + 7 + 31 * frame + salt) % 256);
    }
}

int main(void) {
    /* ---- the label and the status names ------------------------------- */
    const char* label = dlss5_nr_label();
    CHECK(label != NULL);
    CHECK(label != NULL && strstr(label, "synthetic weights") != NULL);
    CHECK(label != NULL && strstr(label, "arbitrary transform") != NULL);
    CHECK(strcmp(dlss5_nr_status_name(DLSS5_NR_OK), "ok") == 0);
    CHECK(dlss5_nr_status_name(DLSS5_NR_INVALID_STRIDE) != NULL);

    /* ---- the rejection matrix (nothing runs; the output stays) -------- */
    {
        dlss5_nr_handle* handle = NULL;
        CHECK_STATUS(dlss5_nr_create(0, kHeight, 8, &handle),
                     DLSS5_NR_INVALID_DIMENSIONS);
        CHECK(handle == NULL);
        CHECK_STATUS(dlss5_nr_create(3000, 3000, 8, &handle),
                     DLSS5_NR_INVALID_DIMENSIONS);
        CHECK(handle == NULL);
        CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 65, &handle),
                     DLSS5_NR_INVALID_VALUES);
        CHECK(handle == NULL);
        CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 8, NULL),
                     DLSS5_NR_NULL_POINTER);
        CHECK_STATUS(dlss5_nr_evaluate(NULL, NULL, 0, NULL, 0, 0),
                     DLSS5_NR_NULL_POINTER);
    }

    /* ---- the sequence: reset, counter schedule, determinism ------------ */
    static unsigned char host[kBytes];
    static unsigned char out[kBytes];
    static unsigned char replay[kBytes];
    static unsigned char snapshot[kBytes];
    static unsigned char sentinel[kBytes];
    dlss5_nr_handle* handle = NULL;
    dlss5_nr_handle* second = NULL;
    CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 8, &handle),
                 DLSS5_NR_OK);
    CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 8, &second), DLSS5_NR_OK);
    CHECK(handle != NULL && second != NULL);
    if (handle == NULL || second == NULL) {
        return 1;
    }
    size_t high_water = 0;
    CHECK_STATUS(dlss5_nr_arena_high_water(handle, &high_water),
                 DLSS5_NR_OK);
    CHECK(high_water == 0);  // nothing ran yet

    for (int frame = 0; frame < 3; ++frame) {
        fill_frame(host, frame, 0);
        memset(out, 0xA5, sizeof(out));
        CHECK_STATUS(dlss5_nr_evaluate(handle, host, 3 * kWidth, out,
                                       3 * kWidth, frame == 0),
                     DLSS5_NR_OK);
        /* The counter: zeroed by the reset at frame 0, then advancing. */
        CHECK(dlss5_nr_counter(handle) == (unsigned int)(frame + 1));
        /* The second handle replays the same stream on fresh state: every
         * frame bit-identical (the pipeline's determinism). */
        CHECK_STATUS(dlss5_nr_evaluate(second, host, 3 * kWidth, replay,
                                       3 * kWidth, frame == 0),
                     DLSS5_NR_OK);
        CHECK(memcmp(out, replay, sizeof(out)) == 0);
        if (frame == 0) {
            memcpy(snapshot, out, sizeof(out));
        }
    }
    /* The arena high-water: reported, positive after the frames and
     * constant across them (the per-frame rewind contract). */
    size_t high_water_2 = 0;
    CHECK_STATUS(dlss5_nr_arena_high_water(handle, &high_water_2),
                 DLSS5_NR_OK);
    CHECK(high_water_2 > 0);

    /* A reset frame: the counter restarts (0 after the reset's commit = 1). */
    fill_frame(host, 3, 0);
    CHECK_STATUS(dlss5_nr_evaluate(handle, host, 3 * kWidth, out,
                                   3 * kWidth, 1),
                 DLSS5_NR_OK);
    CHECK(dlss5_nr_counter(handle) == 1u);
    {
        size_t high_water_3 = 0;
        CHECK_STATUS(dlss5_nr_arena_high_water(handle, &high_water_3),
                     DLSS5_NR_OK);
        CHECK(high_water_3 == high_water_2);
    }

    /* ---- the untouched-output contract on rejection -------------------- */
    memset(sentinel, 0x5A, sizeof(sentinel));
    fill_frame(host, 0, 0);
    CHECK_STATUS(dlss5_nr_evaluate(handle, host, 3 * kWidth - 1, sentinel,
                                   3 * kWidth, 0),
                 DLSS5_NR_INVALID_STRIDE);
    {
        int untouched = 1;
        for (int i = 0; i < kBytes; ++i) {
            if (sentinel[i] != 0x5A) {
                untouched = 0;
            }
        }
        CHECK(untouched);
    }
    CHECK_STATUS(dlss5_nr_evaluate(handle, host, 3 * kWidth, sentinel,
                                   3 * kWidth - 1, 0),
                 DLSS5_NR_INVALID_STRIDE);
    {
        int untouched = 1;
        for (int i = 0; i < kBytes; ++i) {
            if (sentinel[i] != 0x5A) {
                untouched = 0;
            }
        }
        CHECK(untouched);
    }
    CHECK_STATUS(dlss5_nr_evaluate(handle, NULL, 0, out, 0, 0),
                 DLSS5_NR_NULL_POINTER);
    CHECK_STATUS(dlss5_nr_evaluate(handle, host, 0, NULL, 0, 0),
                 DLSS5_NR_NULL_POINTER);

    /* The counter and arena accessors' null contracts. */
    CHECK(dlss5_nr_counter(NULL) == 0u);
    CHECK_STATUS(dlss5_nr_arena_high_water(NULL, &high_water),
                 DLSS5_NR_NULL_POINTER);
    CHECK_STATUS(dlss5_nr_arena_high_water(handle, NULL),
                 DLSS5_NR_NULL_POINTER);

    /* ---- the demonstration-transform mode (NOT DLSS5) --------------- */
    {
        const int w = 64, h = 72, scale = 4;  /* the documented geometry */
        dlss5_nr_handle* demo = NULL;
        CHECK_STATUS(dlss5_nr_create(w, h, 2, &demo), DLSS5_NR_OK);
        /* Mode validation. */
        CHECK_STATUS(dlss5_nr_set_demo_transform(NULL, scale),
                     DLSS5_NR_NULL_POINTER);
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, 1),
                     DLSS5_NR_INVALID_DIMENSIONS);
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, 5),
                     DLSS5_NR_INVALID_DIMENSIONS);
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, scale), DLSS5_NR_OK);
        /* The labels are per-kernel and mandatory EXACTLY (honesty: the
           classical scaler does not claim "real weights"; the neural
           demonstration does). */
        CHECK(strcmp(dlss5_nr_label_for(demo),
                     "not DLSS5 - classical demonstration scaler (Lanczos3)")
              == 0);
        /* The neural demonstration requires real weights loaded first. */
        CHECK(dlss5_nr_demo_model_loaded() == 0);
        CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, DLSS5_NR_DEMO_NEURAL),
                     DLSS5_NR_INVALID_VALUES);
        CHECK_STATUS(dlss5_nr_demo_model_load("does-not-exist.safetensors"),
                     DLSS5_NR_INVALID_VALUES);
        CHECK(dlss5_nr_demo_model_loaded() == 0);
        CHECK_STATUS(dlss5_nr_demo_model_load(NULL), DLSS5_NR_NULL_POINTER);
        /* The model label: a no-model placeholder that still carries the
           in-band "not DLSS5" honesty text. */
        CHECK(dlss5_nr_demo_model_label() != NULL);
        CHECK(strstr(dlss5_nr_demo_model_label(), "not DLSS5") != NULL);
        CHECK(strstr(dlss5_nr_demo_model_label(),
                     "demonstration transform") != NULL);
        CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, DLSS5_NR_DEMO_BILINEAR),
                     DLSS5_NR_OK);
        CHECK(strcmp(dlss5_nr_label_for(demo),
                     "not DLSS5 - classical demonstration scaler "
                     "(bilinear baseline)") == 0);
        CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, DLSS5_NR_DEMO_LANCZOS3),
                     DLSS5_NR_OK);
        CHECK(dlss5_nr_label_for(NULL) == dlss5_nr_label());
        /* One frame: the low-res capture in, scale*w x scale*h out. */
        {
            const int ow = w * scale, oh = h * scale;
            static unsigned char in[64 * 72 * 3];
            static unsigned char out[256 * 288 * 3];
            static unsigned char small_out[64 * 72 * 3];
            int i;
            for (i = 0; i < w * h * 3; ++i) {
                in[i] = (unsigned char)((i * 17) & 0xFF);
            }
            memset(out, 0x7F, sizeof(out));
            memset(small_out, 0x7F, sizeof(small_out));
            /* The output stride must cover the scaled width. */
            CHECK_STATUS(dlss5_nr_evaluate(demo, in, 3 * w, small_out, 3 * w,
                                           1),
                         DLSS5_NR_INVALID_STRIDE);
            CHECK(small_out[0] == 0x7F);  /* untouched on rejection */
            /* The input stride must cover the capture width. */
            CHECK_STATUS(dlss5_nr_evaluate(demo, in, 3 * w - 1, out, 3 * ow,
                                           1),
                         DLSS5_NR_INVALID_STRIDE);
            CHECK_STATUS(dlss5_nr_evaluate(demo, in, 3 * w, out, 3 * ow, 1),
                         DLSS5_NR_OK);
            /* The result must not be the flat untouched sentinel. */
            {
                int changed = 0;
                int j;
                for (j = 0; j < ow * oh * 3; ++j) {
                    if (out[j] != 0x7F) {
                        changed = 1;
                        break;
                    }
                }
                CHECK(changed);
            }
        }
        /* Scales 2 and 3 are accepted (the documented range). */
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, 2), DLSS5_NR_OK);
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, 3), DLSS5_NR_OK);
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, 4), DLSS5_NR_OK);
        /* The A/B kernel setter: validated, and the bilinear kernel
           produces a DIFFERENT result at the same capture (the visible
           A/B's basis) with the same scaled extent. */
        CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, DLSS5_NR_DEMO_BILINEAR),
                     DLSS5_NR_OK);
        {
            /* The transform is at scale 4 here (the last setter call). */
            const int owb = w * 4, ohb = h * 4;
            unsigned char* out_b = (unsigned char*)malloc((size_t)(3 * owb * ohb));
            unsigned char* out_l = (unsigned char*)malloc((size_t)(3 * owb * ohb));
            memset(out_b, 0x7F, (size_t)(3 * owb * ohb));
            memset(out_l, 0x7F, (size_t)(3 * owb * ohb));
            unsigned char* in_b = (unsigned char*)malloc((size_t)(3 * w * h));
            int t;
            for (t = 0; t < 3 * w * h; ++t) {
                in_b[t] = (unsigned char)(t * 251u + 3u);
            }
            CHECK_STATUS(dlss5_nr_evaluate(demo, in_b, 3 * w, out_b, 3 * owb, 1),
                         DLSS5_NR_OK);
            CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, DLSS5_NR_DEMO_LANCZOS3),
                         DLSS5_NR_OK);
            CHECK_STATUS(dlss5_nr_evaluate(demo, in_b, 3 * w, out_l, 3 * owb, 1),
                         DLSS5_NR_OK);
            CHECK(memcmp(out_b, out_l, (size_t)(3 * owb * ohb)) != 0);
            free(in_b);
            free(out_b);
            free(out_l);
        }
        CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, DLSS5_NR_DEMO_LANCZOS3),
                     DLSS5_NR_OK);
        CHECK_STATUS(dlss5_nr_set_demo_kernel(demo, (dlss5_nr_demo_kernel)77),
                     DLSS5_NR_INVALID_VALUES);
        CHECK_STATUS(dlss5_nr_set_demo_kernel(NULL, DLSS5_NR_DEMO_LANCZOS3),
                     DLSS5_NR_NULL_POINTER);
        /* Mode off restores the synthetic pipeline's label. */
        CHECK_STATUS(dlss5_nr_set_demo_transform(demo, 0), DLSS5_NR_OK);

        /* A demo-only handle (extent beyond the network's 320x320 floor):
           creates cleanly, serves the demonstration transform, and the
           synthetic path refuses it with a clean status. */
        {
            dlss5_nr_handle* big = NULL;
            CHECK_STATUS(dlss5_nr_create(640, 400, 2, &big), DLSS5_NR_OK);
            CHECK_STATUS(dlss5_nr_set_demo_transform(big, 2), DLSS5_NR_OK);
            {
                unsigned char* in_big = (unsigned char*)malloc(3u * 640u * 400u);
                unsigned char* out_big = (unsigned char*)malloc(3u * 1280u * 800u);
                int t2;
                for (t2 = 0; t2 < 3 * 640 * 400; ++t2) {
                    in_big[t2] = (unsigned char)(t2 * 197u + 5u);
                }
                CHECK_STATUS(dlss5_nr_evaluate(big, in_big, 3 * 640, out_big,
                                               3 * 1280, 1),
                             DLSS5_NR_OK);
                CHECK_STATUS(dlss5_nr_set_demo_transform(big, 0), DLSS5_NR_OK);
                CHECK_STATUS(dlss5_nr_evaluate(big, in_big, 3 * 640, out_big,
                                               3 * 640, 1),
                             DLSS5_NR_INVALID_DIMENSIONS);
                free(in_big);
                free(out_big);
            }
            dlss5_nr_destroy(big);
        }
        CHECK(dlss5_nr_label_for(demo) == dlss5_nr_label());
        dlss5_nr_destroy(demo);
    }

    dlss5_nr_destroy(handle);
    dlss5_nr_destroy(second);
    dlss5_nr_destroy(NULL);  /* idempotent */

    if (g_failures == 0) {
        printf("native-host: %lld checks passed\n", g_checks);
        return 0;
    }
    printf("native-host: %d check(s) FAILED\n", g_failures);
    return 1;
}
