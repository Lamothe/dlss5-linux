// Native Swin e4m3 linears: the matrix products of the Swin body's MLP on the e4m3
// grid (`_linear` and the `_hidden` expansion+activation in the reference). CPU-only,
// typed, allocation-free; no GPU, headset, proprietary payload, Python or PyTorch
// runtime required.
//
// Semantics adapted from the MIT-licensed reference implementation
// mauri870/dlssnr-pytorch commit 3beef7935a8ef976b47ab78b906e75600c52d2de
// (dlssnr/layers/swin.py "_linear"/"_hidden", dlssnr/int8.py "mm"). The full MIT
// licence notice is retained in vr/NATIVE-SWIN-LINEAR.md.
// Copyright (c) 2026 Mauri de Souza Meneguzzo.
//
// This is an independent implementation of explicitly documented arithmetic semantics.
// It is NOT vendor code, and passing synthetic tests establishes synthetic correctness
// only — never correspondence to NVIDIA's implementation. No proprietary weights,
// captures or binaries are used or implied.

#ifndef DLSS5_NR_NATIVE_SWIN_LINEAR_HPP
#define DLSS5_NR_NATIVE_SWIN_LINEAR_HPP

#include "native-pre-post.hpp"
#include "native-int4.hpp"
#include "native-swin-primitives.hpp"

namespace dlss5_nr {

// ---- documented constants ----------------------------------------------------------------------
inline constexpr int kMaxLinearRows = 1 << 24;  // token rows per call (<= kMaxPixels)
inline constexpr int kMaxLinearDim = 1024;      // operand width/height (4C at C = 256)

// A row-major float32 matrix: `rows` x `cols` values, element (r, c) at
// data + r * row_stride + c, strides in floats with cols <= row_stride.
struct MatrixView {
    const float* data = nullptr;
    int rows = 0;
    int cols = 0;
    std::ptrdiff_t row_stride = 0;
};

struct MutableMatrixView {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;
    std::ptrdiff_t row_stride = 0;
};

// ---- validation shared by the entry points -------------------------------------------------------
// Dimensions, strides and the e4m3 operand grid, in that order, all before any write.
inline Status validate_linear_operand(const MatrixView& m) {
    if (m.data == nullptr) {
        return Status::kNullPointer;
    }
    if (m.rows < 1 || m.rows > kMaxLinearRows || m.cols < 1 || m.cols > kMaxLinearDim) {
        return Status::kInvalidDimensions;
    }
    if (m.row_stride < static_cast<std::ptrdiff_t>(m.cols)
        || m.row_stride > static_cast<std::ptrdiff_t>(kMaxLinearDim)) {
        return Status::kInvalidStride;
    }
    // Every operand value must be finite and exactly on the e4m3 grid: this pins the
    // reference's "e4m3 operands" and makes every product exact in float32.
    for (int r = 0; r < m.rows; ++r) {
        const float* row = m.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(m.row_stride);
        for (int c = 0; c < m.cols; ++c) {
            if (!is_e4m3(row[c])) {
                return Status::kInvalidValues;
            }
        }
    }
    return Status::kOk;
}

inline Status validate_linear_output(const MutableMatrixView* out, int cols,
                                     std::size_t out_capacity) {
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows < 1 || out->rows > kMaxLinearRows || cols < 1 || cols > kMaxLinearDim) {
        return Status::kInvalidDimensions;
    }
    if (out->row_stride < static_cast<std::ptrdiff_t>(cols)
        || out->row_stride > static_cast<std::ptrdiff_t>(kMaxLinearDim)) {
        return Status::kInvalidStride;
    }
    const std::size_t needed = static_cast<std::size_t>(out->rows - 1)
            * static_cast<std::size_t>(out->row_stride)
        + static_cast<std::size_t>(cols);
    if (out_capacity < needed) {
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// ---- the matrix product (`_linear`, default path) ------------------------------------------------
// out [rows, n] = x [rows, k] @ weight [n, k]^T with e4m3-grid operands. The
// reference's default path is a plain matrix product (`int8.mm` with an empty `sites`
// set is `a @ b`; `_linear` contracts `x @ weight.T` for e4m3 operands). This
// implementation makes the float32 accumulator explicit: each product is exact
// (e4m3 significands give at most 8 significant bits), the products are added to a
// float32 accumulator in increasing k order starting at +0.0, and each step rounds
// to float32 (round to nearest, ties to even). The default path states no
// summation-order contract (the phrase "up to float32 summation order" in
// `int8.py` qualifies the optional INT8 W8A8 simulation mode, not this default);
// the k-order choice is this implementation's explicit convention, pinned by the
// accumulation-order test, and is not claimed to match any particular device kernel.
//
// Validation (dimensions, strides, the e4m3 operand grid, output capacity)
// completes before the first output write; on any error `out` is untouched. The
// buffers must not overlap (caller discipline, never rejected).
inline Status swin_linear(const MatrixView& x, const MatrixView& weight,
                          MutableMatrixView* out, std::size_t out_capacity) {
    if (x.data == nullptr || weight.data == nullptr) {
        return Status::kNullPointer;
    }
    if (x.cols != weight.cols) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_operand(x);
        if (status != Status::kOk) {
            return status;
        }
    }
    {
        const Status status = validate_linear_operand(weight);
        if (status != Status::kOk) {
            return status;
        }
    }
    if (out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (out->rows != x.rows) {
        return Status::kInvalidDimensions;
    }
    {
        const Status status = validate_linear_output(out, weight.rows, out_capacity);
        if (status != Status::kOk) {
            return status;
        }
    }
    for (int r = 0; r < x.rows; ++r) {
        const float* x_row = x.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(x.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out->row_stride);
        for (int n = 0; n < weight.rows; ++n) {
            const float* w_row = weight.data + static_cast<std::size_t>(n)
                * static_cast<std::size_t>(weight.row_stride);
            float sum = 0.0f;
            for (int c = 0; c < x.cols; ++c) {
                sum += x_row[c] * w_row[c];
            }
            out_row[n] = sum;
        }
    }
    return Status::kOk;
}

// ---- the e4m3 hidden (`_hidden`, non-int4 path) --------------------------------------------------
// The reference's `_hidden` with no int4 site: `_linear(quantised, expand,
// activate=True)` = `q8(activation(product))` — the cubic SiLU of the product,
// rounded to the e4m3 grid. out [rows, 4C] receives those e4m3 values as float32.
// Same validation and untouched-output contract as swin_linear; `out` must be
// rows x expand.rows.
// The int4 mixed network's per-site overlay (the pinned SwinSite): the
// hard-swish activation parameters (A, B) when the activation changes, and
// the int4 replacement matrices for the expand and QKV projections. A null
// overlay (or a null member) keeps the e4m3 dataflow unchanged.
struct SwinSiteOverlay {
    const float* hs = nullptr;                 // [2] (A, B) or null
    const Int4MatrixView* int4_expand = nullptr;
    const Int4MatrixView* int4_qkv = nullptr;
};

// The int4 expand form of the hidden projection: every x row quantised and
// accumulated through the site matrix into `out` (the same key-ascending
// product contract as the e4m3 matmul's float32 accumulator).
inline Status swin_linear_int4(const MatrixView& x,
                               const Int4MatrixView& matrix,
                               MutableMatrixView* out) {
    if (x.data == nullptr || matrix.q == nullptr || matrix.scale == nullptr
        || matrix.constant == nullptr || matrix.aq == nullptr
        || out == nullptr || out->data == nullptr) {
        return Status::kNullPointer;
    }
    if (matrix.rows != out->cols || matrix.cols != x.cols
        || x.rows != out->rows) {
        return Status::kInvalidDimensions;
    }
    if (matrix.block < 1 || matrix.block > matrix.cols
        || matrix.cols % matrix.block != 0
        || matrix.cols > kMaxLinearDim) {
        return Status::kInvalidValues;
    }
    float codes[kMaxLinearDim];
    for (int r = 0; r < x.rows; ++r) {
        const float* x_row = x.data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(x.row_stride);
        float* out_row = out->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out->row_stride);
        Status status = int4_quantise(matrix, x_row, 1, codes);
        if (status != Status::kOk) {
            return status;
        }
        status = int4_accumulate(matrix, codes, 1, out_row);
        if (status != Status::kOk) {
            return status;
        }
    }
    return Status::kOk;
}

inline Status swin_hidden(const MatrixView& x, const MatrixView& expand,
                          MutableMatrixView* out, std::size_t out_capacity,
                          const SwinSiteOverlay* overlay = nullptr) {
    Status status = Status::kOk;
    const Int4MatrixView* int4_expand = overlay != nullptr
        ? overlay->int4_expand : nullptr;
    if (int4_expand != nullptr) {
        if (out == nullptr || out->data == nullptr) {
            return Status::kNullPointer;
        }
        if (x.rows < 1 || x.cols < 1
            || x.row_stride < x.cols || x.row_stride > kMaxLinearDim) {
            return Status::kInvalidDimensions;
        }
        const Status out_status = validate_linear_output(out, expand.rows,
                                                         out_capacity);
        if (out_status != Status::kOk) {
            return out_status;
        }
        status = swin_linear_int4(x, *int4_expand, out);
    } else {
        status = swin_linear(x, expand, out, out_capacity);
    }
    if (status != Status::kOk) {
        return status;
    }
    const float* hs = overlay != nullptr ? overlay->hs : nullptr;
    for (int r = 0; r < out->rows; ++r) {
        float* out_row = out->data + static_cast<std::size_t>(r)
            * static_cast<std::size_t>(out->row_stride);
        for (int n = 0; n < expand.rows; ++n) {
            out_row[n] = e4m3(hs != nullptr
                                  ? hard_swish_value(out_row[n], hs[0], hs[1])
                                  : swin_activation(out_row[n]));
        }
    }
    return Status::kOk;
}

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_SWIN_LINEAR_HPP
