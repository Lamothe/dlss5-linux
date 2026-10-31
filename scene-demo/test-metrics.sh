#!/usr/bin/env bash
set -euo pipefail

# Standalone CPU-only tests. CXX may select any C++20 compiler.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror \
    "$here/metrics.cpp" -o "$tmp/metrics"
m=$tmp/metrics

# Shell printf emits exact binary bytes, including NUL; all fixtures are analytical.
printf 'P6\n2 2\n255\n\000\000\000\000\000\000\000\000\000\000\000\000' > "$tmp/black.ppm"
printf 'P6\n2 2\n255\n\377\377\377\377\377\377\377\377\377\377\377\377' > "$tmp/white.ppm"
printf 'P6\n2 2\n255\n\000\000\000\377\377\377\000\000\000\377\377\377' > "$tmp/stripes.ppm"
printf 'P6\n# comment\n2 2\n# max\n65535\n\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000' > "$tmp/black16.ppm"

# Exact match: infinite PSNR, unit SSIM and zero errors, including 16-bit input.
"$m" "$tmp/black.ppm" "$tmp/black16.ppm" "$tmp/black.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]
[ "$(grep -c 'SSIM_luma=1 window=1x1' "$tmp/out")" -eq 2 ]
[ "$(grep -c 'detail_gradient_MAE=0' "$tmp/out")" -eq 2 ]
[ "$(grep -c 'lighting_tile_mean_RMSE=0' "$tmp/out")" -eq 2 ]

# Black vs white: RGB MSE=1, SSIM=C1/(1+C1), luma and tile error=1;
# stripes: RGB MSE=.5, gradient error=.5, tile-mean error=.5.
"$m" "$tmp/black.ppm" "$tmp/white.ppm" "$tmp/stripes.ppm" > "$tmp/out"
awk '
function near(a,b) { return (a-b < 0 ? b-a : a-b) < 0.000001 }
/^Cheap path=/ { p=1 } /^Neural path=/ { p=2 }
/PSNR_RGB_dB=/ { split($1,a,"="); if (!near(a[2], p==1 ? 0 : 3.01029995664)) exit 1; n++ }
/SSIM_luma=/ { split($1,a,"="); if (!near(a[2], p==1 ? 0.000099990001 : 0.500049995)) exit 1; n++ }
/luma_MAE=/ { split($1,a,"="); if (!near(a[2], p==1 ? 1 : .5)) exit 1; n++ }
/detail_gradient_MAE=/ { split($1,a,"="); if (!near(a[2], p==1 ? 0 : .5)) exit 1; n++ }
/lighting_tile_mean_RMSE=/ { split($1,a,"="); if (!near(a[2], p==1 ? 1 : .5)) exit 1; n++ }
END { if (n!=10) exit 1 }
' "$tmp/out"

# A single-pixel region isolates the white stripe and has no gradient edges.
"$m" "$tmp/black.ppm" "$tmp/stripes.ppm" "$tmp/black.ppm" --region 1 0 1 1 \
    --diff-cheap "$tmp/diff.ppm" --diff-neural "$tmp/diff-zero.ppm" --sheet "$tmp/sheet.ppm" > "$tmp/out"
grep -q 'region=1,0,1,1' "$tmp/out"
grep -q 'no embedded labels' "$tmp/out"
cmp "$tmp/stripes.ppm" "$tmp/diff.ppm"
cmp "$tmp/black.ppm" "$tmp/diff-zero.ppm"
printf 'P6\n6 2\n255\n\000\000\000\000\000\000\000\000\000\377\377\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\377\377\377\000\000\000\000\000\000' > "$tmp/sheet-expected.ppm"
cmp "$tmp/sheet-expected.ppm" "$tmp/sheet.ppm"

# Valid window is 11x11, or the largest fitting odd square. First raster bytes
# can be whitespace or '#', not skipped as header comments.
{ printf 'P6\r\n11 11\r\n255\r\n'; dd if=/dev/zero bs=363 count=1 status=none; } > "$tmp/large.ppm"
"$m" "$tmp/large.ppm" "$tmp/large.ppm" "$tmp/large.ppm" > "$tmp/out"
grep -q 'SSIM_luma=1 window=11x11' "$tmp/out"
# Uniform black against uniform white gives the analytic Gaussian SSIM C1/(1+C1).
{ printf 'P6\n11 11\n255\n'; head -c 363 /dev/zero | tr '\000' '\377'; } > "$tmp/large-white.ppm"
"$m" "$tmp/large.ppm" "$tmp/large-white.ppm" "$tmp/large.ppm" > "$tmp/out"
awk '/^Cheap path=/ { p=1 } /^Neural path=/ { p=0 } p && /SSIM_luma=/ {
    split($1,a,"="); d=a[2] - 0.000099990001; if (d < -0.000000001 || d > 0.000000001) exit 1; found=1
} END { if (!found) exit 1 }' "$tmp/out"
# Equal average luma but opposite edges: lighting proxy is zero, detail is not.
printf 'P6\n2 2\n255\n\377\377\377\000\000\000\377\377\377\000\000\000' > "$tmp/inverse.ppm"
"$m" "$tmp/stripes.ppm" "$tmp/inverse.ppm" "$tmp/stripes.ppm" > "$tmp/out"
awk '/^Cheap path=/ { p=1 } /^Neural path=/ { p=0 }
p && /detail_gradient_MAE=/ { split($1,a,"="); if (a[2] != 1) exit 1; n++ }
p && /lighting_tile_mean_RMSE=/ { split($1,a,"="); if (a[2] != 0) exit 1; n++ }
END { if (n!=2) exit 1 }' "$tmp/out"
# 16-bit big-endian maxval=256, normalised value 1, agrees with 8-bit white.
# Non-uniform 3x3 Gaussian SSIM, independently evaluated from the checkerboard
# weighted mean: for its inverse var(a)=var(b)=mu*(1-mu), covariance=-var.
printf 'P6\n3 3\n255\n\000\000\000\377\377\377\000\000\000\377\377\377\000\000\000\377\377\377\000\000\000\377\377\377\000\000\000' > "$tmp/checker.ppm"
printf 'P6\n3 3\n255\n\377\377\377\000\000\000\377\377\377\000\000\000\377\377\377\000\000\000\377\377\377\000\000\000\377\377\377' > "$tmp/checker-inverse.ppm"
"$m" "$tmp/checker.ppm" "$tmp/checker-inverse.ppm" "$tmp/checker.ppm" > "$tmp/out"
awk '/^Cheap path=/ { p=1 } /^Neural path=/ { p=0 }
p && /SSIM_luma=/ {
    e=exp(-1/(2*1.5*1.5)); mu=(1+4*e*e)/((1+2*e)*(1+2*e));
    v=mu*(1-mu); expected=((2*mu*(1-mu)+0.0001)*(-2*v+0.0009)) / ((mu*mu+(1-mu)*(1-mu)+0.0001)*(2*v+0.0009));
    split($1,a,"="); d=a[2]-expected; if (d < -0.00000001 || d > 0.00000001) exit 1; found=1
} END { if (!found) exit 1 }' "$tmp/out"
printf 'P6\n1 1\n256\n\001\000\001\000\001\000' > "$tmp/white16.ppm"
printf 'P6\n1 1\n255\n\377\377\377' > "$tmp/white1.ppm"
"$m" "$tmp/white16.ppm" "$tmp/white1.ppm" "$tmp/white1.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]
"$m" "$tmp/large.ppm" "$tmp/large.ppm" "$tmp/large.ppm" --region 0 0 10 8 > "$tmp/out"
grep -q 'SSIM_luma=1 window=7x7' "$tmp/out"
printf 'P6\n1 1\n255\n\043\012\040' > "$tmp/special.ppm"
"$m" "$tmp/special.ppm" "$tmp/special.ppm" "$tmp/special.ppm" > /dev/null


# The expected payload length distinguishes CR + LF raster data from CRLF.
# Compare with unambiguous encodings, not merely each file with itself.
printf 'P6\n1 1\n255\r\012\043\040' > "$tmp/cr-lf-sample.ppm"
printf 'P6\n1 1\n255\r\n\012\043\040' > "$tmp/crlf-lf-sample.ppm"
printf 'P6\n1 1\n255\n\012\043\040' > "$tmp/lf-sample-reference.ppm"
"$m" "$tmp/lf-sample-reference.ppm" "$tmp/cr-lf-sample.ppm" "$tmp/crlf-lf-sample.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]
printf 'P6\n1 1\n65535\r\012\000\000\000\000\000' > "$tmp/cr-lf-sample16.ppm"
printf 'P6\n1 1\n65535\n\012\000\000\000\000\000' > "$tmp/lf-sample16-reference.ppm"
"$m" "$tmp/lf-sample16-reference.ppm" "$tmp/cr-lf-sample16.ppm" "$tmp/cr-lf-sample16.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]

# Only a comment immediately following maxval precedes the raster separator.
# Its terminator is that separator; CRLF is resolved by the exact length.
printf 'P6\n1 1\n255# after max\n\043\012\040' > "$tmp/comment-max-lf.ppm"
printf 'P6\n1 1\n255# after max\r\043\012\040' > "$tmp/comment-max-cr.ppm"
printf 'P6\n1 1\n255# after max\r\n\043\012\040' > "$tmp/comment-max-crlf.ppm"
printf 'P6\n1 1\n255# after max\r\012\043\040' > "$tmp/comment-max-cr-lf-sample.ppm"
"$m" "$tmp/special.ppm" "$tmp/comment-max-lf.ppm" "$tmp/comment-max-cr.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]
"$m" "$tmp/special.ppm" "$tmp/comment-max-crlf.ppm" "$tmp/special.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]
"$m" "$tmp/lf-sample-reference.ppm" "$tmp/comment-max-cr-lf-sample.ppm" "$tmp/lf-sample-reference.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]
# CR-terminated comments in earlier header fields, including CRLF.
printf 'P6\r# magic comment\r1 # width comment\r1\r# max comment\r\n255\n\043\012\040' > "$tmp/comment-header-cr.ppm"
"$m" "$tmp/special.ppm" "$tmp/comment-header-cr.ppm" "$tmp/special.ppm" > "$tmp/out"
[ "$(grep -c 'PSNR_RGB_dB=inf' "$tmp/out")" -eq 2 ]

expect_fail() {
    if "$m" "$@" > "$tmp/out" 2> "$tmp/err"; then
        echo "unexpected success: $*" >&2
        exit 1
    fi
    grep -q '^metrics:' "$tmp/err"
}
# A comment must terminate before, not within, the raster.
printf 'P6\n1 1\n255# unterminated\043\012\040' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/special.ppm" "$tmp/special.ppm"
# Bytes after the separator cannot be consumed as header comments, even
# when doing so would leave exactly the expected three-byte raster.
printf 'P6\n1 1\n255\n#\nABC' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/special.ppm" "$tmp/special.ppm"
printf 'P6\n1 1\n255 # after separator\n\043\012\040' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/special.ppm" "$tmp/special.ppm"
printf 'P3\n1 1\n255\n0 0 0\n' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n1 1\n255\n\000\000' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n1 1\n255\n\000\000\000X' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n1 1\n10\n\000\000\013' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n1 1\n256\n\001\001\000\000\000\000' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n0 1\n255\n' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n18446744073709551615 2\n255\n' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
printf 'P6\n1 1\n65536\n\000\000\000' > "$tmp/bad.ppm"
expect_fail "$tmp/bad.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
expect_fail "$tmp/special.ppm" "$tmp/black.ppm" "$tmp/black.ppm"
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --region 1 0 2 1
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --region 0 0 0 1
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --region -1 0 1 1
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --diff-cheap "$tmp/black.ppm"
ln "$tmp/black.ppm" "$tmp/hardlink.ppm"
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --sheet "$tmp/hardlink.ppm"
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --sheet "$tmp/a.ppm" --diff-cheap "$tmp/a.ppm"
# Report stdout may alias any generated image, including through a hardlink.
# Append avoids shell truncation so an unexpected program write is observable.
printf 'unchanged' > "$tmp/report-alias.ppm"
ln "$tmp/report-alias.ppm" "$tmp/report-hardlink.ppm"
for option in --diff-cheap --diff-neural --sheet; do
    rm -f "$tmp/earlier.ppm"
    earlier=(--diff-cheap "$tmp/earlier.ppm")
    if [ "$option" = --diff-cheap ]; then earlier=(); fi
    if "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
        "${earlier[@]}" "$option" "$tmp/report-hardlink.ppm" \
        >> "$tmp/report-alias.ppm" 2> "$tmp/err"; then
        echo "unexpected success for stdout alias: $option" >&2
        exit 1
    fi
    grep -q '^metrics: cannot exclusively create output:' "$tmp/err"
    [ ! -e "$tmp/earlier.ppm" ]
    [ "$(cat "$tmp/report-alias.ppm")" = unchanged ]
done
if "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
    --diff-cheap "$tmp/report-alias.ppm" >> "$tmp/report-alias.ppm" 2> "$tmp/err"; then
    echo 'unexpected success for direct stdout alias' >&2
    exit 1
fi
grep -q '^metrics: cannot exclusively create output:' "$tmp/err"
[ "$(cat "$tmp/report-alias.ppm")" = unchanged ]
# Existing paths, including dangling symlinks, must never be replaced or followed.
printf 'collision sentinel' > "$tmp/collision.ppm"
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
    --diff-cheap "$tmp/created-before-collision.ppm" --sheet "$tmp/collision.ppm"
[ ! -e "$tmp/created-before-collision.ppm" ]
[ "$(cat "$tmp/collision.ppm")" = 'collision sentinel' ]
ln -s "$tmp/missing-target.ppm" "$tmp/dangling.ppm"
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --sheet "$tmp/dangling.ppm"
[ -L "$tmp/dangling.ppm" ] && [ ! -e "$tmp/missing-target.ppm" ]

# stdout append through a hardlink to any opened input must not modify that input.
for position in 0 1 2; do
    cp "$tmp/black.ppm" "$tmp/stdout-input.ppm"
    rm -f "$tmp/stdout-input-link.ppm"
    ln "$tmp/stdout-input.ppm" "$tmp/stdout-input-link.ppm"
    args=("$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm")
    args[$position]="$tmp/stdout-input.ppm"
    if "$m" "${args[@]}" --sheet "$tmp/stdout-input-output.ppm" \
        >> "$tmp/stdout-input-link.ppm" 2> "$tmp/err"; then
        echo 'unexpected success for stdout input alias' >&2; exit 1
    fi
    grep -q '^metrics: input aliases report output:' "$tmp/err"
    cmp "$tmp/black.ppm" "$tmp/stdout-input.ppm"
    [ ! -e "$tmp/stdout-input-output.ppm" ]
done

# stderr must never corrupt an input, even when report and diagnostic streams
# share that input, or argument parsing fails before the inputs are opened.
for position in 0 1 2; do
    cp "$tmp/black.ppm" "$tmp/stderr-input.ppm"
    rm -f "$tmp/stderr-input-link.ppm"
    ln "$tmp/stderr-input.ppm" "$tmp/stderr-input-link.ppm"
    args=("$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm")
    args[$position]="$tmp/stderr-input.ppm"
    if "$m" "${args[@]}" --bogus > "$tmp/out" 2>> "$tmp/stderr-input-link.ppm"; then
        echo 'unexpected success for stderr input alias' >&2; exit 1
    fi
    cmp "$tmp/black.ppm" "$tmp/stderr-input.ppm"
    if "$m" "${args[@]}" --region 0 0 0 1 > "$tmp/out" 2>> "$tmp/stderr-input-link.ppm"; then
        echo 'unexpected success for stderr input alias during validation' >&2; exit 1
    fi
    cmp "$tmp/black.ppm" "$tmp/stderr-input.ppm"
    if "$m" "${args[@]}" >> "$tmp/stderr-input-link.ppm" 2>&1; then
        echo 'unexpected success for combined input alias' >&2; exit 1
    fi
    cmp "$tmp/black.ppm" "$tmp/stderr-input.ppm"
done
# Usage errors must not write to a named input either.
if "$m" "$tmp/stderr-input.ppm" 2>> "$tmp/stderr-input-link.ppm"; then
    echo 'unexpected success for incomplete arguments' >&2; exit 1
fi
cmp "$tmp/black.ppm" "$tmp/stderr-input.ppm"
# Existing requested outputs are protected as well, even if stderr is separate.
printf 'output sentinel' > "$tmp/stderr-output.ppm"
ln "$tmp/stderr-output.ppm" "$tmp/stderr-output-link.ppm"
if "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
    --sheet "$tmp/stderr-output.ppm" > "$tmp/out" 2>> "$tmp/stderr-output-link.ppm"; then
    echo 'unexpected success for stderr output alias' >&2; exit 1
fi
[ "$(cat "$tmp/stderr-output.ppm")" = 'output sentinel' ]

# Inject an error on the last writable close of the selected inode. Unlike a
# failure on an arbitrary close, this catches writable duplicates that would
# defer the real final-close error until after a success report. O_PATH pins do
# not count as writable handles. The marker proves the injection actually fired.
cat > "$tmp/fail-close.c" <<'CEOF'
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int close(int fd) {
    static int (*real_close)(int);
    if (!real_close) real_close = dlsym(RTLD_NEXT, "close");
    const char *target = getenv("FAIL_LAST_PATH");
    struct stat actual, expected;
    int flags = fcntl(fd, F_GETFL);
    if (target && flags >= 0 && !(flags & O_PATH) && (flags & O_ACCMODE) != O_RDONLY &&
        fstat(fd, &actual) == 0 && stat(target, &expected) == 0 &&
        actual.st_dev == expected.st_dev && actual.st_ino == expected.st_ino &&
        (!getenv("ONLY_STDOUT") || fd == STDOUT_FILENO)) {
        int count = 0;
        DIR *directory = opendir("/proc/self/fd");
        if (directory) {
            struct dirent *entry;
            while ((entry = readdir(directory)) != NULL) {
                char *end;
                long value = strtol(entry->d_name, &end, 10);
                if (*entry->d_name == '\0' || *end != '\0' || value < 0 || value > INT_MAX) continue;
                int candidate = (int)value;
                struct stat other;
                int candidate_flags = fcntl(candidate, F_GETFL);
                if (candidate_flags >= 0 && !(candidate_flags & O_PATH) &&
                    (candidate_flags & O_ACCMODE) != O_RDONLY && fstat(candidate, &other) == 0 &&
                    other.st_dev == actual.st_dev && other.st_ino == actual.st_ino) count++;
            }
            closedir(directory);
            if (count == 1) {
                real_close(fd);  /* Linux consumes the descriptor even when close reports EIO. */
                int marker = open(getenv("MARK"), O_WRONLY | O_CREAT | O_TRUNC, 0600);
                if (marker >= 0) { (void)write(marker, "injected", 8); real_close(marker); }
                errno = EIO;
                return -1;
            }
        }
    }
    return real_close(fd);
}
CEOF
"${CC:-cc}" -shared -fPIC -Wall -Wextra -Werror "$tmp/fail-close.c" -ldl -o "$tmp/fail-close.so"

# Both images have been written before either output is finalised. An error on
# either image's last writable close must remove both, without a success report.
for mode in separate combined; do
    for failed in first second; do
        first="$tmp/finalise-first.ppm"
        second="$tmp/finalise-second.ppm"
        marker="$tmp/close-marker"
        rm -f "$first" "$second" "$marker" "$tmp/report"
        target=$first
        if [ "$failed" = second ]; then target=$second; fi
        if [ "$mode" = combined ]; then
            if env LD_PRELOAD="$tmp/fail-close.so" FAIL_LAST_PATH="$target" MARK="$marker" \
                "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
                --diff-cheap "$first" --sheet "$second" > "$tmp/report" 2>&1; then
                echo "unexpected success on $failed output close ($mode)" >&2; exit 1
            fi
        else
            if env LD_PRELOAD="$tmp/fail-close.so" FAIL_LAST_PATH="$target" MARK="$marker" \
                "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
                --diff-cheap "$first" --sheet "$second" > "$tmp/report" 2> "$tmp/err"; then
                echo "unexpected success on $failed output close ($mode)" >&2; exit 1
            fi
            grep -q '^metrics: cannot close output:' "$tmp/err"
        fi
        [ "$(cat "$marker")" = injected ]
        [ ! -e "$first" ]
        [ ! -e "$second" ]
        ! grep -q '^Reference path=' "$tmp/report"
    done
done

# A last-close error on stdout, with separate or shared stderr, is also fatal.
# The report is provisional and already written, but image outputs must roll
# back; help has no images yet must also report failure on close.
for mode in separate combined help-separate help-combined; do
    first="$tmp/report-close-first.ppm"
    second="$tmp/report-close-second.ppm"
    marker="$tmp/close-marker"
    rm -f "$first" "$second" "$marker"
    : > "$tmp/report"
    args=("$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --diff-cheap "$first" --sheet "$second")
    case "$mode" in help-*) args=(--help) ;; esac
    case "$mode" in
        *combined)
            if env LD_PRELOAD="$tmp/fail-close.so" FAIL_LAST_PATH="$tmp/report" ONLY_STDOUT=1 MARK="$marker" \
                "$m" "${args[@]}" > "$tmp/report" 2>&1; then
                echo "unexpected success on report close ($mode)" >&2; exit 1
            fi
            ! grep -q '^metrics:' "$tmp/report"
            ;;
        *)
            if env LD_PRELOAD="$tmp/fail-close.so" FAIL_LAST_PATH="$tmp/report" ONLY_STDOUT=1 MARK="$marker" \
                "$m" "${args[@]}" > "$tmp/report" 2> "$tmp/err"; then
                echo "unexpected success on report close ($mode)" >&2; exit 1
            fi
            grep -q '^metrics: cannot close report' "$tmp/err"
            ;;
    esac
    [ "$(cat "$marker")" = injected ]
    [ ! -e "$first" ]
    [ ! -e "$second" ]
    case "$mode" in
        help-*) grep -q "existing files and symlinks are never overwritten" "$tmp/report" ;;
        *) grep -q "^Reference path=" "$tmp/report" ;;
    esac
done

# Concurrent contenders for the same initially absent path: exactly one wins.
# The loser cannot truncate the winner's output, regardless of interleaving.
for iteration in $(seq 1 20); do
    target="$tmp/race-$iteration.ppm"
    ("$m" "$tmp/black.ppm" "$tmp/white.ppm" "$tmp/black.ppm" --diff-cheap "$target" > /dev/null 2> "$tmp/race-a.err"; echo $? > "$tmp/race-a.status") &
    a=$!
    ("$m" "$tmp/black.ppm" "$tmp/white.ppm" "$tmp/black.ppm" --diff-cheap "$target" > /dev/null 2> "$tmp/race-b.err"; echo $? > "$tmp/race-b.status") &
    b=$!
    wait "$a" || true; wait "$b" || true
    # With set -e a failing contender exits before writing a status file.
    count=0
    for status in "$tmp/race-a.status" "$tmp/race-b.status"; do
        if [ -f "$status" ]; then [ "$(cat "$status")" = 0 ]; count=$((count + 1)); fi
        rm -f "$status"
    done
    [ "$count" -eq 1 ]
    cmp "$tmp/white.ppm" "$target"
done

# Failure after images were written also removes every owned output.
if "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" \
    --diff-cheap "$tmp/full-diff.ppm" --sheet "$tmp/full-sheet.ppm" > /dev/full 2> "$tmp/err"; then
    echo 'unexpected success writing report to /dev/full' >&2; exit 1
fi
[ ! -e "$tmp/full-diff.ppm" ] && [ ! -e "$tmp/full-sheet.ppm" ]
# Broken stdout pipes are reported as failures rather than terminating before cleanup.
python3 - "$m" "$tmp/black.ppm" "$tmp/broken-pipe.ppm" <<'PYTEST'
import os
import subprocess
import sys

program, image, output = sys.argv[1:]
read_fd, write_fd = os.pipe()
os.close(read_fd)
try:
    result = subprocess.run([program, image, image, image, '--sheet', output],
                            stdout=write_fd, stderr=subprocess.PIPE, check=False)
finally:
    os.close(write_fd)
assert result.returncode != 0, result
assert b'metrics: cannot write report' in result.stderr, result.stderr
assert not os.path.lexists(output)
PYTEST

# An image write exceeding the file-size limit must return an error and clean up.
{ printf 'P6\n30 30\n255\n'; head -c 2700 /dev/zero; } > "$tmp/limit-input.ppm"
if (ulimit -f 1; "$m" "$tmp/limit-input.ppm" "$tmp/limit-input.ppm" "$tmp/limit-input.ppm" \
    --sheet "$tmp/limit-output.ppm" > /dev/null 2> "$tmp/err"); then
    echo 'unexpected success exceeding file-size limit' >&2; exit 1
fi
grep -q '^metrics: cannot write output:' "$tmp/err"
[ ! -e "$tmp/limit-output.ppm" ]

"$m" --help > "$tmp/help"
grep -q 'existing files and symlinks are never overwritten' "$tmp/help"
grep -q 'shell > truncates before metrics can check it' "$tmp/help"
expect_fail "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" --bogus
# Buffered report writes, including --help, must report /dev/full failures.
if "$m" "$tmp/black.ppm" "$tmp/black.ppm" "$tmp/black.ppm" > /dev/full 2> "$tmp/err"; then
    echo 'unexpected success writing report to /dev/full' >&2
    exit 1
fi
grep -q '^metrics: cannot write report' "$tmp/err"
if "$m" --help > /dev/full 2> "$tmp/err"; then
    echo 'unexpected success writing help to /dev/full' >&2
    exit 1
fi
grep -q '^metrics: cannot write report' "$tmp/err"
"$m" --help > /dev/null
printf 'All metrics tests passed.\n'
