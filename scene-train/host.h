#ifndef SCENE_TRAIN_HOST_H
#define SCENE_TRAIN_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stateless, same-resolution RGB colour only: no motion, history or exposure.
   This is a neural rendering demonstration, not DLSS5. */
const char *scene_host_label(void);

typedef struct scene_host scene_host;

/* Only load trusted, locally-produced scene-train model checkpoints from a
   trusted directory; the checkpoint must be a regular file and must not be
   modified while loading. Libtorch archive deserialisation is not safe for untrusted
   files; the file-size limit does not bound decoded storage allocations.
   Loads a separate CPU model per handle. On failure *out is NULL. Error text is
   truncated and NUL-terminated if error is non-NULL and error_capacity > 0.
   Return 0 on success, nonzero on failure. No GPU is used. The error buffer,
   checkpoint string and output-handle storage must not overlap one another.
   Do not pass an existing live handle as *out; destroy it separately. */
int scene_host_load(const char *checkpoint, scene_host **out,
                    char *error, size_t error_capacity);
void scene_host_destroy(scene_host *host);

/* Interleaved RGB8, no colour-space conversion; each row uses 3*width bytes.
   Width and height are 1..16384, with at most 67108864 pixels. Strides and
   buffer sizes are in bytes; padding is not modified. Output is rounded to
   nearest integer (ties to even) after clamping to [0,1]. The output buffer
   is unchanged on failure. Input and output may overlap. Caller must provide
   valid buffers of the specified sizes; the error buffer must not overlap the
   input, output or handle. Serialise calls using one handle, including
   destruction. Separate handles may be used independently. */
int scene_host_infer(scene_host *host, const uint8_t *input, size_t input_size,
                     size_t input_stride, uint8_t *output, size_t output_size,
                     size_t output_stride, uint32_t width, uint32_t height,
                     char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
