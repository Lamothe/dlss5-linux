// Real-ESRGAN x2 (RRDBNet) inference for the demonstration transform.
//
// A self-contained CPU implementation of the RRDBNet scale-2 architecture
// (Xintao Wang et al., "Real-ESRGAN: Training Real-World Blind
// Super-Resolution with Pure Synthetic Data", ICCVW 2021) reading the
// openly released weights from the safetensors container. This is a
// DEMONSTRATION transform with real weights - it is NOT DLSS5 and must
// never be conflated with it. Weights: mlx-community/Real-ESRGAN-x2plus
// (safetensors, F16, 16,703,171 parameters, BSD-3-Clause, converted from
// the official xinntao/Real-ESRGAN release checkpoint).
//
// Architecture (from the upstream BasicSR RRDBNet, scale=2 form):
//   pixel-unshuffle(x, 2) -> conv_first(12->64)
//   -> 23 x RRDB(3 x ResidualDenseBlock(growth 32), residual scale 0.2)
//   -> conv_body + skip
//   -> lrelu(conv_up1(nearest x2)) -> lrelu(conv_up2(nearest x2))
//   -> conv_hr -> conv_last(64->3)
// Input and output are RGB planar float planes in [0,1]; the caller
// clamps. Deterministic (fixed accumulation order per row).

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace demo_rrdb {

struct Tensor {
    std::vector<int> shape;   // e.g. {out, in, 3, 3} (OIHW) or {out}
    std::vector<float> data;  // f32 (converted from the container's dtype)
};

inline float half_to_float(uint16_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0)
    {
        if (mant == 0)
        {
            bits = sign;
        }
        else
        {
            // Subnormal: normalise.
            uint32_t m = mant;
            uint32_t e = 127 - 15 + 1;
            while ((m & 0x400u) == 0)
            {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    }
    else if (exp == 31)
    {
        bits = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// --- safetensors container -------------------------------------------------
// The container: 8-byte little-endian header length, a JSON header
// {tensor_name: {dtype, shape, data_offsets}}, then the raw tensor bytes.
// Only F16 and F32 are accepted (the demonstration needs nothing else).

struct Safetensors {
    std::map<std::string, Tensor> tensors;
    std::string error;  // non-empty on failure

    bool load(const std::string& path)
    {
        tensors.clear();
        error.clear();
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            error = "cannot open weights file: " + path;
            return false;
        }
        uint64_t header_len = 0;
        f.read(reinterpret_cast<char*>(&header_len), 8);
        if (!f || header_len < 2 || header_len > 64u * 1024u * 1024u)
        {
            error = "not a safetensors file (bad header length)";
            return false;
        }
        std::string header(header_len, '\0');
        f.read(header.data(), static_cast<std::streamsize>(header_len));
        if (!f)
        {
            error = "truncated safetensors header";
            return false;
        }
        // The JSON parser: a deliberately small recursive-descent reader
        // (objects, strings, numbers, arrays) - no dependency needed.
        size_t pos = 0;
        if (!parse_header(header, pos))
        {
            return false;
        }
        // Data section.
        f.seekg(0, std::ios::end);
        const auto file_len = static_cast<uint64_t>(f.tellg());
        const uint64_t data_start = 8 + header_len;
        std::vector<char> raw;
        raw.resize(static_cast<size_t>(file_len - data_start));
        f.seekg(static_cast<std::streamsize>(data_start));
        f.read(raw.data(), static_cast<std::streamsize>(raw.size()));
        if (!f)
        {
            error = "truncated safetensors data section";
            return false;
        }
        for (auto& kv : ext_)
        {
            const TensorExt& e = kv.second;
            Tensor& t = tensors[kv.first];
            t.shape = e.shape;
            const size_t count = shape_count(e.shape);
            if (count == 0 && !e.shape.empty() && shape_count(e.shape) == 0)
            {
                error = "tensor '" + kv.first + "' has a bad shape";
                return false;
            }
            // Wrap-safe: division arithmetic and a width check.
            if ((e.width != 2 && e.width != 4) ||
                e.offset_end < e.offset_begin ||
                e.offset_end > raw.size() ||
                (e.offset_end - e.offset_begin) % e.width != 0 ||
                (e.offset_end - e.offset_begin) / e.width != count ||
                count > raw.size())
            {
                error = "tensor '" + kv.first + "' has a mismatched data range";
                return false;
            }
            t.data.resize(count);
            const char* srcp = raw.data() + e.offset_begin;
            if (e.width == 2)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    uint16_t hv;
                    std::memcpy(&hv, srcp + i * 2, 2);
                    t.data[i] = half_to_float(hv);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    float v;
                    std::memcpy(&v, srcp + i * 4, 4);
                    t.data[i] = v;
                }
            }
        }
        ext_.clear();
        return true;
    }

    // Parsed-tensor fields held transiently during load.
    struct RawEntry {
        std::string dtype;
        std::vector<int> shape;
        uint64_t offset_begin = 0;
        uint64_t offset_end = 0;
    };

private:
    static size_t shape_count(const std::vector<int>& shape)
    {
        size_t n = 1;
        for (int s : shape)
        {
            n *= static_cast<size_t>(s);
        }
        return n;
    }

    // Extend the tensor with the offsets/dtype; the element width lives on
    // the tensor during load only.
    struct TensorExt : Tensor {
        uint64_t offset_begin = 0;
        uint64_t offset_end = 0;
        size_t width = 0;
    };

    std::map<std::string, TensorExt> ext_;

    // The JSON reader (subset).
    std::string js_;
    size_t jp_ = 0;

    void jskip()
    {
        while (jp_ < js_.size() && (js_[jp_] == ' ' || js_[jp_] == '\n' ||
                                    js_[jp_] == '\t' || js_[jp_] == '\r'))
        {
            jp_++;
        }
    }

    bool jstring(std::string& out)
    {
        jskip();
        if (jp_ >= js_.size() || js_[jp_] != '"')
        {
            error = "malformed safetensors JSON (string expected)";
            return false;
        }
        jp_++;
        out.clear();
        while (jp_ < js_.size() && js_[jp_] != '"')
        {
            if (js_[jp_] == '\\')
            {
                jp_++;
                if (jp_ >= js_.size())
                {
                    error = "malformed safetensors JSON (escape)";
                    return false;
                }
                const char c = js_[jp_++];
                out += (c == 'n') ? '\n' : (c == 't') ? '\t' : c;
            }
            else
            {
                out += js_[jp_++];
            }
        }
        if (jp_ >= js_.size())
        {
            error = "malformed safetensors JSON (unterminated string)";
            return false;
        }
        jp_++;  // closing quote
        return true;
    }

    bool jnumber(double& out)
    {
        jskip();
        const char* start = js_.c_str() + jp_;
        char* end = nullptr;
        out = strtod(start, &end);
        if (end == start)
        {
            error = "malformed safetensors JSON (number expected)";
            return false;
        }
        jp_ += static_cast<size_t>(end - start);
        return true;
    }

    bool jvalue_tensor(const std::string& name)
    {
        jskip();
        if (jp_ >= js_.size() || js_[jp_] != '{')
        {
            error = "malformed safetensors JSON (object expected)";
            return false;
        }
        jp_++;
        RawEntry e;
        while (true)
        {
            jskip();
            if (jp_ < js_.size() && js_[jp_] == '}')
            {
                jp_++;
                break;
            }
            std::string key;
            if (!jstring(key))
            {
                return false;
            }
            jskip();
            if (jp_ >= js_.size() || js_[jp_] != ':')
            {
                error = "malformed safetensors JSON (':' expected)";
                return false;
            }
            jp_++;
            if (key == "dtype")
            {
                if (!jstring(e.dtype))
                {
                    return false;
                }
            }
            else if (key == "shape")
            {
                jskip();
                if (jp_ >= js_.size() || js_[jp_] != '[')
                {
                    error = "malformed safetensors JSON (shape array)";
                    return false;
                }
                jp_++;
                while (true)
                {
                    jskip();
                    if (jp_ < js_.size() && js_[jp_] == ']')
                    {
                        jp_++;
                        break;
                    }
                    double v;
                    if (!jnumber(v))
                    {
                        return false;
                    }
                    if (!(v >= 0.0 && v <= 1048576.0))
                    {
                        error = "safetensors: bad tensor dimension";
                        return false;
                    }
                    e.shape.push_back(static_cast<int>(v));
                    jskip();
                    if (jp_ < js_.size() && js_[jp_] == ',')
                    {
                        jp_++;
                    }
                }
            }
            else if (key == "data_offsets")
            {
                jskip();
                if (jp_ >= js_.size() || js_[jp_] != '[')
                {
                    error = "malformed safetensors JSON (offsets array)";
                    return false;
                }
                jp_++;
                double v;
                if (!jnumber(v))
                {
                    return false;
                }
                if (!(v >= 0.0 && v <= 4.0e18))
                {
                    error = "safetensors: bad data offset";
                    return false;
                }
                e.offset_begin = static_cast<uint64_t>(v);
                jskip();
                if (jp_ < js_.size() && js_[jp_] == ',')
                {
                    jp_++;
                }
                if (!jnumber(v))
                {
                    return false;
                }
                if (!(v >= 0.0 && v <= 4.0e18))
                {
                    error = "safetensors: bad data offset";
                    return false;
                }
                e.offset_end = static_cast<uint64_t>(v);
                jskip();
                if (jp_ >= js_.size() || js_[jp_] != ']')
                {
                    error = "malformed safetensors JSON (offsets array end)";
                    return false;
                }
                jp_++;
            }
            else
            {
                // Unknown key: skip a scalar or array.
                jskip();
                if (jp_ < js_.size() && js_[jp_] == '[')
                {
                    int depth = 0;
                    do
                    {
                        if (js_[jp_] == '[')
                        {
                            depth++;
                        }
                        if (js_[jp_] == ']')
                        {
                            depth--;
                        }
                        jp_++;
                    } while (jp_ < js_.size() && depth > 0);
                }
                else if (jp_ < js_.size() && js_[jp_] == '"')
                {
                    std::string dump;
                    if (!jstring(dump))
                    {
                        return false;
                    }
                }
                else
                {
                    double dump;
                    if (!jnumber(dump))
                    {
                        return false;
                    }
                }
            }
            jskip();
            if (jp_ < js_.size() && js_[jp_] == ',')
            {
                jp_++;
            }
        }
        if (name == "__metadata__")
        {
            return true;
        }
        size_t elem_width = 0;
        if (e.dtype == "F16")
        {
            elem_width = 2;
        }
        else if (e.dtype == "F32")
        {
            elem_width = 4;
        }
        else
        {
            error = "tensor '" + name + "': unsupported dtype " + e.dtype +
                    " (only F16/F32 are accepted)";
            return false;
        }
        TensorExt t;
        t.shape = e.shape;
        t.offset_begin = e.offset_begin;
        t.offset_end = e.offset_end;
        t.width = elem_width;
        ext_[name] = t;
        return true;
    }

    bool parse_header(const std::string& header, size_t)
    {
        js_ = header;
        jp_ = 0;
        ext_.clear();
        jskip();
        if (jp_ >= js_.size() || js_[jp_] != '{')
        {
            error = "malformed safetensors JSON (top-level object)";
            return false;
        }
        jp_++;
        while (true)
        {
            jskip();
            if (jp_ < js_.size() && js_[jp_] == '}')
            {
                jp_++;
                break;
            }
            std::string name;
            if (!jstring(name))
            {
                return false;
            }
            jskip();
            if (jp_ >= js_.size() || js_[jp_] != ':')
            {
                error = "malformed safetensors JSON (':' expected)";
                return false;
            }
            jp_++;
            if (!jvalue_tensor(name))
            {
                return false;
            }
            jskip();
            if (jp_ < js_.size() && js_[jp_] == ',')
            {
                jp_++;
            }
        }
        for (auto& kv : ext_)
        {
            tensors[kv.first] = kv.second;
        }
        return true;
    }
};

// --- RRDBNet x2 -----------------------------------------------------------

class RRDBNet {
public:
    struct Conv {
        const Tensor* w = nullptr;  // [out, in, 3, 3]
        const Tensor* b = nullptr;  // [out]
        int out_ch = 0;
        int in_ch = 0;
    };

    // Load and verify the topology: every expected tensor name and shape,
    // nothing missing, nothing mismatched. The error text names the exact
    // mismatch (the weights-interface contract).
    bool load(const std::string& path)
    {
        error_.clear();
        // Atomic: a failed load (parse OR verification) must leave any
        // previously loaded model intact (the conv pointers reference the
        // tensor map, so both are backed up).
        Safetensors backup_st = std::move(st_);
        auto backup_convs = std::move(convs_);
        bool committed = false;
        using ConvMap = std::map<std::string, Conv>;
        struct Restore
        {
            Safetensors* st;
            ConvMap* convs;
            Safetensors* backup_st;
            ConvMap* backup_convs;
            bool* committed;
            ~Restore()
            {
                if (!*committed)
                {
                    *st = std::move(*backup_st);
                    *convs = std::move(*backup_convs);
                }
            }
        } guard{&st_, &convs_, &backup_st, &backup_convs, &committed};
        if (!st_.load(path))
        {
            error_ = st_.error;
            return false;
        }
        // Topology: scale-2 RRDBNet, 23 blocks, 64 features, growth 32.
        const int blocks = 23;
        auto need = [&](const std::string& name,
                        std::initializer_list<int> shape) -> const Tensor* {
            auto it = st_.tensors.find(name);
            if (it == st_.tensors.end())
            {
                error_ = "missing tensor: " + name;
                return nullptr;
            }
            const Tensor& t = it->second;
            if (t.shape.size() != shape.size())
            {
                error_ = "tensor " + name + ": rank mismatch";
                return nullptr;
            }
            size_t i = 0;
            for (int s : shape)
            {
                if (t.shape[i] != s)
                {
                    std::ostringstream os;
                    os << "tensor " << name << ": shape mismatch (expected "
                       << s << " at dim " << i << ", found " << t.shape[i]
                       << ")";
                    error_ = os.str();
                    return nullptr;
                }
                ++i;
            }
            return &t;
        };
        auto conv = [&](const std::string& name, int out, int in) -> bool {
            // MLX weight layout: [out, kh, kw, in].
            const Tensor* w = need(name + ".weight", {out, 3, 3, in});
            const Tensor* b = need(name + ".bias", {out});
            if (!w || !b)
            {
                return false;
            }
            convs_[name] = Conv{w, b, out, in};
            return true;
        };
        if (!conv("conv_first", 64, 12) || !conv("conv_body", 64, 64) ||
            !conv("conv_up1", 64, 64) || !conv("conv_up2", 64, 64) ||
            !conv("conv_hr", 64, 64) || !conv("conv_last", 3, 64))
        {
            return false;
        }
        for (int i = 0; i < blocks; ++i)
        {
            for (int j = 1; j <= 3; ++j)  // RRDB holds rdb1..rdb3
            {
                for (int k = 1; k <= 5; ++k)  // each RDB holds conv1..conv5
                {
                    std::ostringstream os;
                    os << "body." << i << ".rdb" << j << ".conv" << k;
                    const int out = (k == 5) ? 64 : 32;
                    const int in = 64 + (k - 1) * 32;
                    if (!conv(os.str(), out, in))
                    {
                        return false;
                    }
                }
            }
        }
        // Nothing unexpected: the container must hold exactly the expected
        // tensors (a mismatched/foreign file is rejected, not half-used).
        const size_t expected = 6 * 2 + blocks * 3 * 5 * 2;
        if (st_.tensors.size() != expected)
        {
            std::ostringstream os;
            os << "tensor count mismatch: expected " << expected << ", found "
               << st_.tensors.size();
            error_ = os.str();
            return false;
        }
        loaded_ = true;
        committed = true;
        return true;
    }

    bool loaded() const { return loaded_; }
    const std::string& error() const { return error_; }
    int param_count() const
    {
        int n = 0;
        for (const auto& kv : st_.tensors)
        {
            int c = 1;
            for (int s : kv.second.shape)
            {
                c *= s;
            }
            n += c;
        }
        return n;
    }

    // forward_x2: RGB planar input [0,1] at (w,h) -> RGB planar output at
    // (2w, 2h). Deterministic.
    void forward_x2(const float* in_r, const float* in_g, const float* in_b,
                    int w, int h, float* out_r, float* out_g, float* out_b,
                    int threads)
    {
        // 1. pixel-unshuffle x2: (w,h,3) -> (w/2,h/2,12).
        const int uw = w / 2;
        const int uh = h / 2;
        std::vector<float> un(static_cast<size_t>(12) * uw * uh);
        pixel_unshuffle2(in_r, in_g, in_b, w, h, un.data(), threads);
        // 2. conv_first 12->64 + trunk.
        std::vector<float> feat(static_cast<size_t>(64) * uw * uh);
        conv3x3(un.data(), 12, feat.data(), 64, uw, uh, "conv_first", threads);
        std::vector<float> body = feat;
        for (int i = 0; i < 23; ++i)
        {
            rrdb(body.data(), uw, uh, i, threads);
        }
        std::vector<float> body_out(static_cast<size_t>(64) * uw * uh);
        conv3x3(body.data(), 64, body_out.data(), 64, uw, uh, "conv_body",
                threads);
        for (size_t i = 0; i < feat.size(); ++i)
        {
            feat[i] += body_out[i];
        }
        // 3. Upsample: nearest x2 twice with conv_up1/conv_up2.
        std::vector<float> up1(static_cast<size_t>(64) * uw * 2 * uh * 2);
        nearest2x(feat.data(), 64, uw, uh, up1.data());
        std::vector<float> up1c(up1.size());
        conv3x3(up1.data(), 64, up1c.data(), 64, uw * 2, uh * 2, "conv_up1",
                threads);
        lrelu_inplace(up1c);
        std::vector<float> up2(up1.size() * 4);
        nearest2x(up1c.data(), 64, uw * 2, uh * 2, up2.data());
        std::vector<float> up2c(up2.size());
        conv3x3(up2.data(), 64, up2c.data(), 64, uw * 4, uh * 4, "conv_up2",
                threads);
        lrelu_inplace(up2c);
        // 4. conv_hr + conv_last -> RGB.
        std::vector<float> hr(up2c.size());
        conv3x3(up2c.data(), 64, hr.data(), 64, uw * 4, uh * 4, "conv_hr",
                threads);
        lrelu_inplace(hr);
        std::vector<float> out(static_cast<size_t>(3) * uw * 4 * uh * 4);
        conv3x3(hr.data(), 64, out.data(), 3, uw * 4, uh * 4, "conv_last",
                threads);
        // 5. Split to planar output (the extent is 2w x 2h exactly).
        const int ow = uw * 4;
        const int oh = uh * 4;
        const size_t plane = static_cast<size_t>(ow) * oh;
        for (size_t i = 0; i < plane; ++i)
        {
            out_r[i] = out[i];
            out_g[i] = out[plane + i];
            out_b[i] = out[2 * plane + i];
        }
    }

private:
    Safetensors st_;
    std::map<std::string, Conv> convs_;
    std::string error_;
    bool loaded_ = false;

    Conv conv_of(const std::string& name) const
    {
        auto it = convs_.find(name);
        return it != convs_.end() ? it->second : Conv{};
    }

    static void lrelu_inplace(std::vector<float>& v)
    {
        for (auto& x : v)
        {
            if (x < 0)
            {
                x *= 0.2f;
            }
        }
    }

    // Channel-major (NCHW-style) planar buffers throughout.
    void conv3x3(const float* in, int in_ch, float* out, int out_ch, int w,
                 int h, const std::string& name, int threads) const
    {
        const Conv c = conv_of(name);
        const float* wgt = c.w->data.data();
        const float* bias = c.b->data.data();
        const size_t plane = static_cast<size_t>(w) * h;
        const int nthreads = threads < 1 ? 1 : threads;
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t)
        {
            pool.emplace_back([&, t]() {
                // Zero-padded 3x3 (upstream conv3x3 semantics): taps
                // outside the image contribute 0, never a clamped
                // sample. The interior columns run through a contiguous,
                // vectorisable x loop; the border rows/columns skip
                // their out-of-bounds taps.
                for (int oc = t; oc < out_ch; oc += nthreads)
                {
                    float* op = out + static_cast<size_t>(oc) * plane;
                    for (size_t i = 0; i < plane; ++i)
                    {
                        op[i] = bias[oc];
                    }
                    for (int ic = 0; ic < in_ch; ++ic)
                    {
                        const float* ip =
                            in + static_cast<size_t>(ic) * plane;
                        // MLX weight layout [out, kh, kw, in]: one
                        // output filter is 9*in_ch floats.
                        const float* kw = wgt +
                            static_cast<size_t>(oc) * 9 * in_ch;
                        for (int y = 0; y < h; ++y)
                        {
                            float* dp = op + y * w;
                            for (int ky = 0; ky < 3; ++ky)
                            {
                                const int yy = y + ky - 1;
                                if (yy < 0 || yy >= h)
                                {
                                    continue;
                                }
                                const float* row = ip +
                                    static_cast<size_t>(yy) * w;
                                const float k0 = kw[(ky * 3 + 0) * in_ch + ic];
                                const float k1 = kw[(ky * 3 + 1) * in_ch + ic];
                                const float k2 = kw[(ky * 3 + 2) * in_ch + ic];
                                for (int x = 1; x < w - 1; ++x)
                                {
                                    dp[x] += row[x - 1] * k0 + row[x] * k1 +
                                             row[x + 1] * k2;
                                }
                            }
                            // The two edge columns (zero-pad skips).
                            for (int e = 0; e < (w == 1 ? 1 : 2); ++e)
                            {
                                const int x = e == 0 ? 0 : w - 1;
                                float sum = 0.0f;
                                for (int ky = 0; ky < 3; ++ky)
                                {
                                    const int yy = y + ky - 1;
                                    if (yy < 0 || yy >= h)
                                    {
                                        continue;
                                    }
                                    for (int kx = 0; kx < 3; ++kx)
                                    {
                                        const int xx = x + kx - 1;
                                        if (xx < 0 || xx >= w)
                                        {
                                            continue;
                                        }
                                        sum += ip[static_cast<size_t>(yy) * w
                                                  + xx] *
                                               kw[(ky * 3 + kx) * in_ch + ic];
                                    }
                                }
                                dp[x] += sum;
                            }
                        }
                    }
                }
            });
        }
        for (auto& th : pool)
        {
            th.join();
        }
    }

    void rrdb(float* feat, int w, int h, int block, int threads) const
    {
        const size_t plane = static_cast<size_t>(w) * h;
        std::vector<float> x(feat, feat + 64 * plane);
        for (int j = 1; j <= 3; ++j)
        {
            // ResidualDenseBlock: five convs with growing concatenations.
            std::vector<float> c1(32 * plane), c2(32 * plane),
                c3(32 * plane), c4(32 * plane), c5(64 * plane);
            std::ostringstream n1, n2, n3, n4, n5;
            n1 << "body." << block << ".rdb" << j << ".conv1";
            n2 << "body." << block << ".rdb" << j << ".conv2";
            n3 << "body." << block << ".rdb" << j << ".conv3";
            n4 << "body." << block << ".rdb" << j << ".conv4";
            n5 << "body." << block << ".rdb" << j << ".conv5";
            conv3x3(x.data(), 64, c1.data(), 32, w, h, n1.str(), threads);
            lrelu_inplace(c1);
            {
                std::vector<float> cat(static_cast<size_t>(96) * plane);
                std::memcpy(cat.data(), x.data(), 64 * plane * sizeof(float));
                std::memcpy(cat.data() + 64 * plane, c1.data(),
                            32 * plane * sizeof(float));
                conv3x3(cat.data(), 96, c2.data(), 32, w, h, n2.str(),
                        threads);
                lrelu_inplace(c2);
            }
            {
                std::vector<float> cat(static_cast<size_t>(128) * plane);
                std::memcpy(cat.data(), x.data(), 64 * plane * sizeof(float));
                std::memcpy(cat.data() + 64 * plane, c1.data(),
                            32 * plane * sizeof(float));
                std::memcpy(cat.data() + 96 * plane, c2.data(),
                            32 * plane * sizeof(float));
                conv3x3(cat.data(), 128, c3.data(), 32, w, h, n3.str(),
                        threads);
                lrelu_inplace(c3);
            }
            {
                std::vector<float> cat(static_cast<size_t>(160) * plane);
                std::memcpy(cat.data(), x.data(), 64 * plane * sizeof(float));
                std::memcpy(cat.data() + 64 * plane, c1.data(),
                            32 * plane * sizeof(float));
                std::memcpy(cat.data() + 96 * plane, c2.data(),
                            32 * plane * sizeof(float));
                std::memcpy(cat.data() + 128 * plane, c3.data(),
                            32 * plane * sizeof(float));
                conv3x3(cat.data(), 160, c4.data(), 32, w, h, n4.str(),
                        threads);
                lrelu_inplace(c4);
            }
            {
                std::vector<float> cat(static_cast<size_t>(192) * plane);
                std::memcpy(cat.data(), x.data(), 64 * plane * sizeof(float));
                std::memcpy(cat.data() + 64 * plane, c1.data(),
                            32 * plane * sizeof(float));
                std::memcpy(cat.data() + 96 * plane, c2.data(),
                            32 * plane * sizeof(float));
                std::memcpy(cat.data() + 128 * plane, c3.data(),
                            32 * plane * sizeof(float));
                std::memcpy(cat.data() + 160 * plane, c4.data(),
                            32 * plane * sizeof(float));
                conv3x3(cat.data(), 192, c5.data(), 64, w, h, n5.str(),
                        threads);
            }
            // x5 * 0.2 + x.
            for (size_t i = 0; i < x.size(); ++i)
            {
                x[i] += c5[i] * 0.2f;
            }
        }
        // RRDB residual: out * 0.2 + feat.
        for (size_t i = 0; i < 64 * plane; ++i)
        {
            feat[i] += x[i] * 0.2f;
        }
    }

    static void pixel_unshuffle2(const float* r, const float* g,
                                 const float* b, int w, int h, float* out,
                                 int)
    {
        // (w,h,3) -> (w/2,h/2,12) channel-major: out[c*uh*uw + y*uw + x].
        const int uw = w / 2;
        const int uh = h / 2;
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const int ox = x / 2;
                const int oy = y / 2;
                const int sub = (y & 1) * 2 + (x & 1);  // 0..3
                const float* src[3] = {r, g, b};
                for (int c = 0; c < 3; ++c)
                {
                    out[(c * 4 + sub) * uh * uw + oy * uw + ox] =
                        src[c][y * w + x];
                }
            }
        }
    }

    static void nearest2x(const float* in, int ch, int w, int h, float* out)
    {
        const int ow = w * 2;
        const int oh = h * 2;
        for (int c = 0; c < ch; ++c)
        {
            const float* ip = in + static_cast<size_t>(c) * w * h;
            float* op = out + static_cast<size_t>(c) * ow * oh;
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    const float v = ip[y * w + x];
                    op[(2 * y) * ow + 2 * x] = v;
                    op[(2 * y) * ow + 2 * x + 1] = v;
                    op[(2 * y + 1) * ow + 2 * x] = v;
                    op[(2 * y + 1) * ow + 2 * x + 1] = v;
                }
            }
        }
    }
};

}  // namespace demo_rrdb
