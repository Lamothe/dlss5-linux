/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Michael Lamothe
 *
 * The Milestone 2 desktop stereo dry run (vr/VR-SEAM-M1-ALIGNMENT.md): two
 * evaluator handles as the two per-eye contexts the Genesis contract's ABI
 * requires (neural-seam.h: one context per eye, never shared), fed synthetic
 * stereo pairs at demo-ish geometry. Pins per-eye history INDEPENDENCE (a
 * shared history must fail), history consumption, the per-handle counter
 * schedule, the reset semantics the rule-2 discontinuity composition relies
 * on (the composition itself is pinned by the seam adapter suite), per-eye
 * determinism, and writes a side-by-side evidence artefact with the in-band
 * label recorded beside it.
 *
 * No headset and no renderer: the inputs are synthetic stereo (the right eye
 * is the left shifted by a fixed parallax), which is exactly what the dry run
 * needs — the eye plumbing, not scene correctness. Compiled as C and linked
 * against the C++ implementation (the ABI exactly as a host sees it), built
 * by vr/test-native-stereo.sh.
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

enum {
    kWidth = 64,
    kHeight = 72,
    kBytes = 3 * kWidth * kHeight,
    kParallax = 4, /* pixels; the synthetic stereo baseline */
    kSeparator = 6 /* pixels of grey between the two eyes in the artefact */
};

/* Synthetic per-eye input: a deterministic pattern whose phase encodes the
 * frame; the right eye is the same pattern shifted by kParallax (a real
 * horizontal disparity, so the side-by-side artefact reads as stereo). */
static void fill_eye(unsigned char* rgb, int frame, int eye) {
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            int sx = x + (eye == 1 ? kParallax : 0);
            int p = (y * 3 + sx * 5 + 37 * frame) % 256;
            int i = 3 * (y * kWidth + x);
            rgb[i + 0] = (unsigned char)p;
            rgb[i + 1] = (unsigned char)((p * 3 + 11) % 256);
            rgb[i + 2] = (unsigned char)((p * 7 + 29) % 256);
        }
    }
}

static void fill_frame_pair(unsigned char* left, unsigned char* right,
                            int frame) {
    fill_eye(left, frame, 0);
    fill_eye(right, frame, 1);
}

/* The side-by-side artefact: every write is checked so a truncated file on
 * a full disk can never masquerade as evidence. */
static int write_side_by_side(const char* path,
                              const unsigned char* left,
                              const unsigned char* right) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        return 0;
    }
    int width = 2 * kWidth + kSeparator;
    int ok = fprintf(f, "P6\n%d %d\n255\n", width, kHeight) > 0;
    for (int y = 0; ok && y < kHeight; ++y) {
        ok = ok && fwrite(left + 3 * y * kWidth, 1, (size_t)(3 * kWidth), f)
            == (size_t)(3 * kWidth);
        for (int x = 0; ok && x < kSeparator; ++x) {
            unsigned char grey[3] = {40, 40, 40};
            ok = ok && fwrite(grey, 1, 3, f) == 3;
        }
        ok = ok && fwrite(right + 3 * y * kWidth, 1, (size_t)(3 * kWidth), f)
            == (size_t)(3 * kWidth);
    }
    ok = ok && fflush(f) == 0 && ferror(f) == 0;
    return fclose(f) == 0 && ok;
}

int main(int argc, char** argv) {
    const char* out_dir = argc > 1 ? argv[1] : NULL;

    /* ---- the label travels with any presented output ------------------ */
    const char* label = dlss5_nr_label();
    CHECK(label != NULL);
    CHECK(label != NULL && strstr(label, "synthetic weights") != NULL);
    CHECK(label != NULL && strstr(label, "arbitrary transform") != NULL);

    dlss5_nr_handle* left = NULL;
    dlss5_nr_handle* right = NULL;
    dlss5_nr_handle* fresh_l = NULL;
    dlss5_nr_handle* fresh_r = NULL;
    dlss5_nr_handle* late = NULL;
    unsigned char* in_l = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* in_r = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* out_l = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* out_r = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* ref_l = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* ref_r = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* tmp_l = (unsigned char*)malloc((size_t)kBytes);
    unsigned char* tmp_r = (unsigned char*)malloc((size_t)kBytes);
    CHECK(in_l != NULL && in_r != NULL && out_l != NULL && out_r != NULL
          && ref_l != NULL && ref_r != NULL && tmp_l != NULL && tmp_r != NULL);
    if (g_failures != 0) {
        goto cleanup; /* no evaluation may run on a failed allocation */
    }

    /* ---- frame 1: the first frame's reset convention (rule 1) --------- */
    CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 4, &left), DLSS5_NR_OK);
    CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 4, &right), DLSS5_NR_OK);
    if (left == NULL || right == NULL) {
        goto cleanup;
    }
    fill_frame_pair(in_l, in_r, 1);
    CHECK_STATUS(dlss5_nr_evaluate(left, in_l, 3 * kWidth, out_l, 3 * kWidth, 1),
                 DLSS5_NR_OK);
    CHECK_STATUS(
        dlss5_nr_evaluate(right, in_r, 3 * kWidth, out_r, 3 * kWidth, 1),
        DLSS5_NR_OK);
    CHECK(dlss5_nr_counter(left) == 1u);
    CHECK(dlss5_nr_counter(right) == 1u);
    memcpy(ref_l, out_l, (size_t)kBytes);
    memcpy(ref_r, out_r, (size_t)kBytes);

    /* ---- frame 2: history continues, counters advance per handle ------ */
    fill_frame_pair(in_l, in_r, 2);
    CHECK_STATUS(dlss5_nr_evaluate(left, in_l, 3 * kWidth, out_l, 3 * kWidth, 0),
                 DLSS5_NR_OK);
    CHECK_STATUS(
        dlss5_nr_evaluate(right, in_r, 3 * kWidth, out_r, 3 * kWidth, 0),
        DLSS5_NR_OK);
    CHECK(dlss5_nr_counter(left) == 2u);
    CHECK(dlss5_nr_counter(right) == 2u);

    /* History is actually consumed: the frame-2 result must differ from
     * the same input evaluated with a reset (zero history). */
    CHECK_STATUS(dlss5_nr_evaluate(left, in_l, 3 * kWidth, tmp_l, 3 * kWidth, 1),
                 DLSS5_NR_OK);
    CHECK(memcmp(tmp_l, out_l, (size_t)kBytes) != 0);
    CHECK(dlss5_nr_counter(left) == 1u); /* the reset restarted the schedule */

    /* ---- per-eye history independence ---------------------------------
     * Both eyes now hold DIFFERENT history (frames 1-2 shifted vs
     * unshifted). The same input through both contexts must produce
     * different outputs (a shared history would make them identical), and
     * each must differ from a fresh context's result for the same input.
     * The deterministic synthetic pipeline makes these exact comparisons. */
    fill_eye(in_l, 7, 0); /* identical bytes for both eyes */
    CHECK_STATUS(dlss5_nr_evaluate(left, in_l, 3 * kWidth, out_l, 3 * kWidth, 0),
                 DLSS5_NR_OK);
    CHECK_STATUS(
        dlss5_nr_evaluate(right, in_l, 3 * kWidth, out_r, 3 * kWidth, 0),
        DLSS5_NR_OK);
    CHECK(memcmp(out_l, out_r, (size_t)kBytes) != 0);
    CHECK_STATUS(
        dlss5_nr_create(kWidth, kHeight, 4, &fresh_l), DLSS5_NR_OK);
    if (fresh_l != NULL) {
        CHECK_STATUS(
            dlss5_nr_evaluate(fresh_l, in_l, 3 * kWidth, tmp_l, 3 * kWidth, 0),
            DLSS5_NR_OK);
        CHECK(memcmp(out_l, tmp_l, (size_t)kBytes) != 0);
        CHECK(memcmp(out_r, tmp_l, (size_t)kBytes) != 0);
    }

    /* A context created mid-sequence starts its own schedule (a handle's
     * history is its own: contract "never reused across features"). */
    CHECK_STATUS(dlss5_nr_create(kWidth, kHeight, 4, &late), DLSS5_NR_OK);
    if (late != NULL) {
        fill_frame_pair(in_l, in_r, 9);
        CHECK_STATUS(
            dlss5_nr_evaluate(late, in_l, 3 * kWidth, tmp_l, 3 * kWidth, 1),
            DLSS5_NR_OK);
        CHECK(dlss5_nr_counter(late) == 1u);
        /* frames since each handle's last reset: left ran frame1, frame2,
         * then a reset call (schedule restarted) and the identical frame
         * -> 2; right ran frame1, frame2 and the identical frame with no
         * reset -> 3. Exact values, so a shared counter cannot hide. */
        CHECK(dlss5_nr_counter(left) == 2u);
        CHECK(dlss5_nr_counter(right) == 3u);
    }
    dlss5_nr_destroy(late);
    late = NULL;

    /* ---- reset determinism: frame 1 with reset == 1 reproduces frame 1 */
    fill_frame_pair(in_l, in_r, 1);
    CHECK_STATUS(dlss5_nr_evaluate(left, in_l, 3 * kWidth, tmp_l, 3 * kWidth, 1),
                 DLSS5_NR_OK);
    CHECK_STATUS(
        dlss5_nr_evaluate(right, in_r, 3 * kWidth, tmp_r, 3 * kWidth, 1),
        DLSS5_NR_OK);
    CHECK(memcmp(tmp_l, ref_l, (size_t)kBytes) == 0);
    CHECK(memcmp(tmp_r, ref_r, (size_t)kBytes) == 0);
    CHECK(dlss5_nr_counter(left) == 1u);
    CHECK(dlss5_nr_counter(right) == 1u);

    /* ---- the reset semantics the discontinuity composition relies on --
     * Contract rule 2 composes an epoch change as reset ORed into both
     * eyes; the composition itself is pinned by the seam adapter suite.
     * Here: reset == 1 on a used context must equal the fresh-history
     * result for the same frame (so the rule-2 OR produces a clean
     * history), on both eyes. */
    fill_frame_pair(in_l, in_r, 3);
    CHECK_STATUS(dlss5_nr_evaluate(left, in_l, 3 * kWidth, out_l, 3 * kWidth, 1),
                 DLSS5_NR_OK);
    CHECK_STATUS(
        dlss5_nr_evaluate(right, in_r, 3 * kWidth, out_r, 3 * kWidth, 1),
        DLSS5_NR_OK);
    CHECK_STATUS(
        dlss5_nr_create(kWidth, kHeight, 4, &fresh_r), DLSS5_NR_OK);
    if (fresh_r != NULL) {
        CHECK_STATUS(
            dlss5_nr_evaluate(fresh_r, in_r, 3 * kWidth, tmp_r, 3 * kWidth, 1),
            DLSS5_NR_OK);
        CHECK(memcmp(out_r, tmp_r, (size_t)kBytes) == 0);
    }

    /* ---- the side-by-side evidence artefact --------------------------- */
    if (out_dir != NULL) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/stereo-side-by-side.ppm", out_dir);
        CHECK(write_side_by_side(path, out_l, out_r) != 0);
        snprintf(path, sizeof(path), "%s/report.txt", out_dir);
        FILE* report = fopen(path, "w");
        CHECK(report != NULL);
        if (report != NULL) {
            fprintf(report, "stereo dry run (two evaluator contexts)\n");
            fprintf(report, "label: %s\n", label);
            fprintf(report, "geometry: %dx%d per eye (padded to the 320 floor)\n",
                    kWidth, kHeight);
            fprintf(report, "parallax: %d px synthetic baseline\n", kParallax);
            fprintf(report, "artefact: stereo-side-by-side.ppm (%dx%d, "
                            "left | grey | right)\n",
                    2 * kWidth + kSeparator, kHeight);
            fprintf(report, "mode: frame-by-frame inspect; CPU evaluator, "
                            "seconds per frame, never real-time\n");
            fprintf(report, "result: %s (%lld checks)\n",
                    g_failures == 0 ? "PASS" : "FAIL", g_checks);
            fclose(report);
        }
    }

    dlss5_nr_destroy(left);
    dlss5_nr_destroy(right);
    dlss5_nr_destroy(fresh_l);
    dlss5_nr_destroy(fresh_r);
    left = NULL;
    right = NULL;
    fresh_l = NULL;
    fresh_r = NULL;

cleanup:
    dlss5_nr_destroy(left);
    dlss5_nr_destroy(right);
    dlss5_nr_destroy(fresh_l);
    dlss5_nr_destroy(fresh_r);
    dlss5_nr_destroy(late);
    free(in_l);
    free(in_r);
    free(out_l);
    free(out_r);
    free(ref_l);
    free(ref_r);
    free(tmp_l);
    free(tmp_r);

    if (g_failures == 0) {
        printf("native-stereo: %lld checks passed\n", g_checks);
        return 0;
    }
    printf("native-stereo: %d check(s) FAILED\n", g_failures);
    return 1;
}
