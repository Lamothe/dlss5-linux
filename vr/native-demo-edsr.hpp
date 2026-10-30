// EDSR-base x2 (Lim et al., "Enhanced Deep Residual Networks for Single
// Image Super-Resolution", CVPRW 2017) for the demonstration transform.
//
// A self-contained CPU implementation reading the openly released
// super-image checkpoint (eugenesiow/edsr-base, Apache-2.0, DIV2K,
// pytorch_model_2x.pt) from the torch-save container (a ZIP of a pickle
// stream plus raw storages). EDSR is a CLASSICAL super-resolution model
// (L1-trained on clean degradation). No quality claim is made here: the
// measured numbers, including where this model underperforms classical
// scalers against a pristine ground truth, are in NATIVE-DEMO-UPSCALE.md.
// NOT DLSS5 - a demonstration transform with real weights.
//
// Forward (the upstream super-image implementation):
//   x = head(x)               conv3x3 3->64
//   res = body(x)             16 x ResBlock(conv-ReLU-conv, res_scale=1)
//                             + conv3x3 64->64
//   res += x                  global residual
//   x = tail(res)             conv3x3 64->256 + PixelShuffle(2)
//                             + conv3x3 64->3
// The MeanShift buffers in the checkpoint are unused by the upstream
// forward (verified against the source); the harness measures both
// variants and the report names the one used.

#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "native-demo-rrdbnet.hpp"  // Tensor + half_to_float

namespace demo_edsr {

using demo_rrdb::Tensor;

// --- torch-save container (ZIP of a pickle stream + raw storages) ---------
// Only the state-dict subset is supported: an OrderedDict of str -> tensor
// rebuilt by torch._utils._rebuild_tensor_v2 over persistent storages.
// Anything else is rejected with a clear error (the weights-interface
// contract).

class TorchStateDict {
public:
    std::map<std::string, Tensor> tensors;
    std::string error;

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
        f.seekg(0, std::ios::end);
        const auto len = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> zip(len);
        f.read(zip.data(), static_cast<std::streamsize>(len));
        if (!f || len < 22)
        {
            error = "truncated torch-save file";
            return false;
        }
        // ZIP entries are read from the CENTRAL DIRECTORY (torch's
        // writer sets the data-descriptor flag, leaving the local
        // header sizes zero; the directory carries the real sizes).
        std::map<std::string, std::vector<char>> entries;
        // EOCD: scan back for the signature (no zip64 for model files).
        long eocd = -1;
        for (long i = static_cast<long>(len) - 22; i >= 0 && i + 22 > static_cast<long>(len) - 22 - 65558; --i)
        {
            if (std::memcmp(zip.data() + i, "PK\x05\x06", 4) == 0)
            {
                eocd = i;
                break;
            }
        }
        if (eocd < 0)
        {
            error = "not a zip container (no end-of-central-directory)";
            return false;
        }
        const uint16_t entry_count = read16(zip, static_cast<size_t>(eocd) + 10);
        const uint32_t cd_offset = read32(zip, static_cast<size_t>(eocd) + 16);
        long cd = static_cast<long>(cd_offset);
        for (uint16_t e = 0; e < entry_count; ++e)
        {
            if (cd + 46 > static_cast<long>(len) ||
                std::memcmp(zip.data() + cd, "PK\x01\x02", 4) != 0)
            {
                error = "corrupt central directory";
                return false;
            }
            const uint16_t method = read16(zip, static_cast<size_t>(cd) + 10);
            const uint32_t comp = read32(zip, static_cast<size_t>(cd) + 20);
            const uint16_t nlen = read16(zip, static_cast<size_t>(cd) + 28);
            const uint16_t xlen = read16(zip, static_cast<size_t>(cd) + 30);
            const uint16_t clen = read16(zip, static_cast<size_t>(cd) + 32);
            const size_t local = read32(zip, static_cast<size_t>(cd) + 42);
            if (static_cast<size_t>(cd) + 46 + nlen + xlen + clen > len)
            {
                error = "corrupt central directory entry";
                return false;
            }
            std::string name(zip.data() + cd + 46, nlen);
            if (method != 0)
            {
                error = "compressed zip entry '" + name +
                        "' is not supported (torch-save stores raw)";
                return false;
            }
            // The data starts after the LOCAL header's own name/extra
            // fields (their lengths can differ from the directory's).
            if (local > len || local + 30 > len)
            {
                error = "corrupt zip local header";
                return false;
            }
            const uint16_t lnlen = read16(zip, static_cast<size_t>(local) + 26);
            const uint16_t lxlen = read16(zip, static_cast<size_t>(local) + 28);
            const size_t data_at = local + 30 + lnlen + lxlen;
            if (data_at < local || data_at + comp > len || comp > len)
            {
                error = "corrupt zip entry '" + name + "'";
                return false;
            }
            entries[name] = std::vector<char>(
                zip.begin() + static_cast<long>(data_at),
                zip.begin() + static_cast<long>(data_at + comp));
            cd += 46 + nlen + xlen + clen;
        }
        auto pkl = entries.find("archive/data.pkl");
        if (pkl == entries.end())
        {
            error = "not a torch-save file (missing archive/data.pkl)";
            return false;
        }
        if (!parse_pickle(pkl->second, entries))
        {
            return false;
        }
        return true;
    }

private:
    static uint16_t read16(const std::vector<char>& v, size_t p)
    {
        return static_cast<uint8_t>(v[p]) |
               (static_cast<uint16_t>(static_cast<uint8_t>(v[p + 1])) << 8);
    }

    // Bounds check for a k-byte read at p in an n-byte stream.
    static bool need(size_t p, size_t k, size_t n, std::string& error)
    {
        if (p + k > n || p + k < p)
        {
            error = "pickle: truncated stream";
            return false;
        }
        return true;
    }
    static uint32_t read32(const std::vector<char>& v, size_t p)
    {
        return static_cast<uint32_t>(static_cast<uint8_t>(v[p])) |
               (static_cast<uint32_t>(static_cast<uint8_t>(v[p + 1])) << 8) |
               (static_cast<uint32_t>(static_cast<uint8_t>(v[p + 2])) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(v[p + 3])) << 24);
    }

    // The pickle VM: a stack of tagged values sufficient for the
    // state-dict subset.
    struct Value {
        enum Kind { None, Int, Float, Str, Tuple, StorageRef } kind = None;
        long i = 0;
        double d = 0;
        std::string s;
        std::vector<Value> t;
        int storage = -1;   // the storage index (StorageRef)
        long offset = 0, numel = 0;
        std::vector<long> size, stride;
    };

    std::vector<Value> stack_;
    std::map<long, Value> memo_;
    long memo_next_ = 0;
    std::vector<std::string> globals_;
    std::map<std::string, std::vector<char>> entries_;
    std::map<int, Tensor> storages_;

    bool parse_pickle(const std::vector<char>& pkl,
                      const std::map<std::string, std::vector<char>>& entries)
    {
        entries_ = entries;
        stack_.clear();
        memo_.clear();
        size_t p = 0;
        const size_t n = pkl.size();
        bool done = false;
        auto pop = [&](Value& out) -> bool {
            if (stack_.empty())
            {
                error = "pickle: stack underflow";
                return false;
            }
            out = stack_.back();
            stack_.pop_back();
            return true;
        };
        while (p < n && !done)
        {
            const uint8_t op = static_cast<uint8_t>(pkl[p++]);
            Value v;
            switch (op)
            {
                case 0x80:  // PROTO
                    ++p;
                    break;
                case 0x63:  // GLOBAL: module\nname\n
                {
                    std::string mod, name;
                    while (p < n && pkl[p] != '\n')
                    {
                        mod += pkl[p++];
                    }
                    ++p;
                    while (p < n && pkl[p] != '\n')
                    {
                        name += pkl[p++];
                    }
                    ++p;
                    v.kind = Value::Str;
                    v.s = mod + "." + name;
                    stack_.push_back(v);
                    break;
                }
                case 0x71:  // BINPUT
                {
                    if (!need(p, 1, n, error))
                    {
                        return false;
                    }
                    const long k = static_cast<uint8_t>(pkl[p++]);
                    if (!stack_.empty())
                    {
                        memo_[k] = stack_.back();
                    }
                    break;
                }
                case 0x72:  // LONG_BINPUT
                {
                    if (!need(p, 4, n, error))
                    {
                        return false;
                    }
                    const long k = read32(pkl, p);
                    p += 4;
                    if (!stack_.empty())
                    {
                        memo_[k] = stack_.back();
                    }
                    break;
                }
                case 0x68:  // BINGET
                {
                    if (!need(p, 1, n, error))
                    {
                        return false;
                    }
                    const long k = static_cast<uint8_t>(pkl[p++]);
                    auto it = memo_.find(k);
                    if (it == memo_.end())
                    {
                        error = "pickle: bad memo reference";
                        return false;
                    }
                    stack_.push_back(it->second);
                    break;
                }
                case 0x6a:  // LONG_BINGET
                {
                    if (!need(p, 4, n, error))
                    {
                        return false;
                    }
                    const long k = read32(pkl, p);
                    p += 4;
                    auto it = memo_.find(k);
                    if (it == memo_.end())
                    {
                        error = "pickle: bad memo reference";
                        return false;
                    }
                    stack_.push_back(it->second);
                    break;
                }
                case 0x58:  // BINUNICODE
                {
                    if (!need(p, 4, n, error))
                    {
                        return false;
                    }
                    const uint32_t l = read32(pkl, p);
                    p += 4;
                    if (!need(p, l, n, error))
                    {
                        return false;
                    }
                    v.kind = Value::Str;
                    v.s.assign(pkl.data() + p, l);
                    p += l;
                    stack_.push_back(v);
                    break;
                }
                case 0x8c:  // SHORT_BINUNICODE
                {
                    if (!need(p, 1, n, error))
                    {
                        return false;
                    }
                    const uint8_t l = static_cast<uint8_t>(pkl[p++]);
                    if (!need(p, l, n, error))
                    {
                        return false;
                    }
                    v.kind = Value::Str;
                    v.s.assign(pkl.data() + p, l);
                    p += l;
                    stack_.push_back(v);
                    break;
                }
                case 0x4b:  // BININT1
                    if (!need(p, 1, n, error))
                    {
                        return false;
                    }
                    v.kind = Value::Int;
                    v.i = static_cast<uint8_t>(pkl[p++]);
                    stack_.push_back(v);
                    break;
                case 0x4d:  // BININT2
                    if (!need(p, 2, n, error))
                    {
                        return false;
                    }
                    v.kind = Value::Int;
                    v.i = read16(pkl, p);
                    p += 2;
                    stack_.push_back(v);
                    break;
                case 0x4a:  // BININT
                    if (!need(p, 4, n, error))
                    {
                        return false;
                    }
                    v.kind = Value::Int;
                    v.i = static_cast<int32_t>(read32(pkl, p));
                    p += 4;
                    stack_.push_back(v);
                    break;
                case 0x8a:  // LONG1
                {
                    if (!need(p, 1, n, error))
                    {
                        return false;
                    }
                    const uint8_t l = static_cast<uint8_t>(pkl[p++]);
                    if (l > sizeof(long) || !need(p, l, n, error))
                    {
                        error = "pickle: bad LONG1 length";
                        return false;
                    }
                    // Two's-complement little-endian with sign extension
                    // from the top byte (the pickle spec).
                    unsigned long uval = 0;
                    for (uint8_t i = 0; i < l; ++i)
                    {
                        uval |= static_cast<unsigned long>(
                                    static_cast<uint8_t>(pkl[p + i]))
                                << (8 * i);
                    }
                    if (l > 0 && (static_cast<uint8_t>(pkl[p + l - 1]) & 0x80))
                    {
                        for (uint8_t i = l; i < sizeof(long); ++i)
                        {
                            uval |= 0xFFUL << (8 * i);
                        }
                    }
                    p += l;
                    const long val = static_cast<long>(uval);
                    v.kind = Value::Int;
                    v.i = val;
                    stack_.push_back(v);
                    break;
                }
                case 0x47:  // BINFLOAT (big-endian double)
                {
                    if (!need(p, 8, n, error))
                    {
                        return false;
                    }
                    char tmp[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        tmp[i] = pkl[p + (7 - i)];
                    }
                    double d;
                    std::memcpy(&d, tmp, 8);
                    p += 8;
                    v.kind = Value::Float;
                    v.d = d;
                    stack_.push_back(v);
                    break;
                }
                case 0x29:  // EMPTY_TUPLE
                    v.kind = Value::Tuple;
                    stack_.push_back(v);
                    break;
                case 0x28:  // MARK
                    v.kind = Value::None;
                    v.i = -1;  // the marker tag
                    stack_.push_back(v);
                    break;
                case 0x74:  // TUPLE (up to the mark)
                case 0x85:  // TUPLE1
                case 0x86:  // TUPLE2
                case 0x87:  // TUPLE3
                {
                    std::vector<Value> items;
                    if (op != 0x74)
                    {
                        const size_t k = op == 0x85 ? 1 : (op == 0x86 ? 2 : 3);
                        if (stack_.size() < k)
                        {
                            error = "pickle: stack underflow";
                            return false;
                        }
                        items.insert(items.end(), stack_.end() -
                                     static_cast<long>(k), stack_.end());
                        stack_.resize(stack_.size() - k);
                    }
                    else
                    {
                        long idx = static_cast<long>(stack_.size()) - 1;
                        while (idx >= 0 &&
                               !(stack_[idx].kind == Value::None &&
                                 stack_[idx].i == -1))
                        {
                            --idx;
                        }
                        if (idx < 0)
                        {
                            error = "pickle: tuple without mark";
                            return false;
                        }
                        items.insert(items.end(), stack_.begin() + idx + 1,
                                     stack_.end());
                        stack_.resize(static_cast<size_t>(idx));
                    }
                    v.kind = Value::Tuple;
                    v.t = items;
                    stack_.push_back(v);
                    break;
                }
                case 0x88:  // NEWTRUE
                case 0x89:  // NEWFALSE
                    v.kind = Value::Int;
                    v.i = op == 0x88 ? 1 : 0;
                    stack_.push_back(v);
                    break;
                case 0x4e:  // NONE
                    stack_.push_back(v);
                    break;
                case 0x51:  // BINPERSID: the storage reference tuple
                {
                    Value tup;
                    if (!pop(tup))
                    {
                        return false;
                    }
                    v.kind = Value::StorageRef;
                    if (tup.kind == Value::Tuple && tup.t.size() >= 5)
                    {
                        // The persistent id: ("storage", storage_type,
                        // key, location, numel) - the key is the decimal
                        // storage index ("0", "1", ...).
                        if (tup.t[2].kind == Value::Str)
                        {
                            v.storage = std::atoi(tup.t[2].s.c_str());
                        }
                        else
                        {
                            v.storage = static_cast<int>(tup.t[2].i);
                        }
                        v.i = tup.t[4].i;  // the storage numel
                    }
                    stack_.push_back(v);
                    break;
                }
                case 0x52:  // REDUCE
                {
                    Value args, fn;
                    if (!pop(args) || !pop(fn))
                    {
                        return false;
                    }
                    if (fn.kind == Value::Str &&
                        fn.s.find("_rebuild_tensor") != std::string::npos)
                    {
                        // (storage, offset, size, stride, requires_grad,
                        //  backward_hooks)
                        if (args.kind != Value::Tuple || args.t.size() < 4)
                        {
                            error = "pickle: bad tensor rebuild args";
                            return false;
                        }
                        v.kind = Value::Str;
                        v.s = "#tensor";
                        v.storage = static_cast<int>(args.t[0].storage);
                        v.offset = args.t[1].i;
                        v.numel = args.t.empty() ? 0 : args.t[0].i;
                        v.size.clear();
                        v.stride.clear();
                        for (const auto& s : args.t[2].t)
                        {
                            v.size.push_back(s.i);
                        }
                        for (const auto& s : args.t[3].t)
                        {
                            v.stride.push_back(s.i);
                        }
                        stack_.push_back(v);
                    }
                    else if (fn.kind == Value::Str &&
                             (fn.s.find("OrderedDict") != std::string::npos ||
                              fn.s.find("dict") != std::string::npos))
                    {
                        v.kind = Value::Tuple;  // an empty container
                        stack_.push_back(v);
                    }
                    else
                    {
                        error = "pickle: unsupported callable " +
                                (fn.kind == Value::Str ? fn.s : "?");
                        return false;
                    }
                    break;
                }
                case 0x73:  // SETITEM: dict[key] = value
                {
                    Value value, key, dict;
                    if (!pop(value) || !pop(key))
                    {
                        return false;
                    }
                    if (value.kind == Value::Str && value.s == "#tensor" &&
                        key.kind == Value::Str)
                    {
                        if (!materialise(key.s, value))
                        {
                            return false;
                        }
                    }
                    // The dict itself stays on the stack (ignore nested
                    // non-tensor entries such as the hooks container).
                    break;
                }
                case 0x7d:  // EMPTY_DICT
                    v.kind = Value::Tuple;  // an empty container
                    stack_.push_back(v);
                    break;
                case 0x75:  // SETITEMS: dict.update(pairs up to the mark)
                {
                    long idx = static_cast<long>(stack_.size()) - 1;
                    while (idx >= 0 &&
                           !(stack_[idx].kind == Value::None &&
                             stack_[idx].i == -1))
                    {
                        --idx;
                    }
                    if (idx < 0)
                    {
                        error = "pickle: setitems without mark";
                        return false;
                    }
                    // Pairs above the mark (k1, v1, k2, v2, ...).
                    for (long i = idx + 1; i + 1 < static_cast<long>(stack_.size());
                         i += 2)
                    {
                        const Value& key = stack_[i];
                        const Value& value = stack_[i + 1];
                        if (value.kind == Value::Str && value.s == "#tensor" &&
                            key.kind == Value::Str)
                        {
                            if (!materialise(key.s, value))
                            {
                                return false;
                            }
                        }
                    }
                    stack_.resize(static_cast<size_t>(idx));
                    break;
                }
                case 0x62:  // BUILD: obj.__setstate__(state) - the state
                            // carries no tensors for the state-dict subset.
                {
                    Value state, obj;
                    if (!pop(state) || !pop(obj))
                    {
                        return false;
                    }
                    stack_.push_back(obj);
                    break;
                }
                case 0x2e:  // STOP
                    done = true;
                    break;
                default:
                    error = "pickle: unsupported opcode " + std::to_string(op);
                    return false;
            }
        }
        if (tensors.empty())
        {
            error = "no tensors decoded from the state dict";
            return false;
        }
        return true;
    }

    bool materialise(const std::string& name, const Value& t)
    {
        // The storage: archive/data/<index>, raw f32 (torch default).
        auto it = entries_.find("archive/data/" + std::to_string(t.storage));
        if (it == entries_.end())
        {
            error = "missing storage for tensor " + name;
            return false;
        }
        const auto& raw = it->second;
        Tensor out;
        out.shape.assign(t.size.begin(), t.size.end());
        // Wrap-safe: every dim is in [0, 2^20], the product and the
        // offset range are checked by division arithmetic.
        size_t count = 1;
        for (long s : t.size)
        {
            if (s <= 0 || s > (1L << 20) ||
                count > (static_cast<size_t>(1) << 50) / static_cast<size_t>(s))
            {
                error = "bad shape for tensor " + name;
                return false;
            }
            count *= static_cast<size_t>(s);
        }
        if (t.offset < 0)
        {
            error = "bad storage offset for tensor " + name;
            return false;
        }
        const size_t total = raw.size() / 4;
        const size_t off = static_cast<size_t>(t.offset);
        if (off > total || count > total - off)
        {
            error = "storage too small for tensor " + name;
            return false;
        }
        out.data.resize(count);
        const char* src = raw.data() + off * 4;
        for (size_t i = 0; i < count; ++i)
        {
            float v;
            std::memcpy(&v, src + i * 4, 4);
            out.data[i] = v;
        }
        tensors[name] = out;
        return true;
    }
};

// --- EDSR-base x2 forward -------------------------------------------------

class EDSR {
public:
    bool load(const std::string& path)
    {
        error_.clear();
        // Atomic: a failed load (parse OR verification) must leave any
        // previously loaded model intact - back up and restore on the
        // failure paths.
        TorchStateDict backup = std::move(st_);
        bool committed = false;
        struct Restore
        {
            TorchStateDict* st;
            TorchStateDict* backup;
            bool* committed;
            ~Restore()
            {
                if (!*committed)
                {
                    *st = std::move(*backup);
                }
            }
        } guard{&st_, &backup, &committed};
        if (!st_.load(path))
        {
            error_ = st_.error;
            return false;
        }
        // Topology: 16 ResBlocks (64 feats), head + body tail conv +
        // tail upsample. Every expected name and shape; nothing else.
        auto need = [&](const std::string& name, int rows, int cols, int in,
                        int out) -> bool {
            auto it = st_.tensors.find(name);
            if (it == st_.tensors.end())
            {
                error_ = "missing tensor: " + name;
                return false;
            }
            const Tensor& t = it->second;
            if (t.shape.size() != 4 || t.shape[0] != out || t.shape[1] != in ||
                t.shape[2] != rows || t.shape[3] != cols)
            {
                error_ = "tensor " + name + ": shape mismatch";
                return false;
            }
            return true;
        };
        auto bias = [&](const std::string& name, int out) -> bool {
            auto it = st_.tensors.find(name);
            if (it == st_.tensors.end())
            {
                error_ = "missing tensor: " + name;
                return false;
            }
            if (it->second.shape.size() != 1 || it->second.shape[0] != out)
            {
                error_ = "tensor " + name + ": shape mismatch";
                return false;
            }
            return true;
        };
        if (!need("module.head.0.weight", 3, 3, 3, 64) ||
            !bias("module.head.0.bias", 64) ||
            !need("module.body.16.weight", 3, 3, 64, 64) ||
            !bias("module.body.16.bias", 64) ||
            !need("module.tail.0.0.weight", 3, 3, 64, 256) ||
            !bias("module.tail.0.0.bias", 256) ||
            !need("module.tail.1.weight", 3, 3, 64, 3) ||
            !bias("module.tail.1.bias", 3))
        {
            return false;
        }
        for (int i = 0; i < 16; ++i)
        {
            const std::string base = "module.body." + std::to_string(i) + ".body.";
            if (!need(base + "0.weight", 3, 3, 64, 64) ||
                !bias(base + "0.bias", 64) ||
                !need(base + "2.weight", 3, 3, 64, 64) ||
                !bias(base + "2.bias", 64))
            {
                return false;
            }
        }
        // The MeanShift buffers exist in the checkpoint and are verified
        // but unused by the upstream forward (see the header comment).
        if (st_.tensors.size() != 76)
        {
            error_ = "tensor count mismatch: expected 76, found " +
                     std::to_string(st_.tensors.size());
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

    // forward_x2: RGB planar [0,1] at (w,h) -> RGB planar [0,1] at (2w,2h).
    // The model operates in [0,255] (rgb_range=255); the conversion is at
    // the boundary. Deterministic.
    void forward_x2(const float* in_r, const float* in_g, const float* in_b,
                    int w, int h, float* out_r, float* out_g, float* out_b,
                    int threads)
    {
        const size_t plane = static_cast<size_t>(w) * h;
        std::vector<float> x(3 * plane);
        for (size_t i = 0; i < plane; ++i)
        {
            x[i] = in_r[i] * 255.0f;
            x[plane + i] = in_g[i] * 255.0f;
            x[2 * plane + i] = in_b[i] * 255.0f;
        }
        std::vector<float> head(64 * plane);
        conv3x3(x.data(), 3, head.data(), 64, w, h, "module.head.0",
                threads);
        std::vector<float> body = head;
        for (int i = 0; i < 16; ++i)
        {
            const std::string base =
                "module.body." + std::to_string(i) + ".body.";
            std::vector<float> r1(64 * plane);
            conv3x3(body.data(), 64, r1.data(), 64, w, h, base + "0", threads);
            for (auto& v : r1)
            {
                if (v < 0)
                {
                    v = 0;
                }
            }
            std::vector<float> r2(64 * plane);
            conv3x3(r1.data(), 64, r2.data(), 64, w, h, base + "2", threads);
            for (size_t j = 0; j < body.size(); ++j)
            {
                body[j] += r2[j];  // res_scale = 1
            }
        }
        std::vector<float> bt(64 * plane);
        conv3x3(body.data(), 64, bt.data(), 64, w, h, "module.body.16",
                threads);
        for (size_t i = 0; i < head.size(); ++i)
        {
            head[i] += bt[i];  // global residual
        }
        // tail: conv 64->256 + PixelShuffle(2) + conv 64->3.
        std::vector<float> up(256 * plane);
        conv3x3(head.data(), 64, up.data(), 256, w, h, "module.tail.0.0",
                threads);
        const int ow = w * 2;
        const int oh = h * 2;
        std::vector<float> shuf(64 * static_cast<size_t>(ow) * oh);
        for (int y = 0; y < oh; ++y)
        {
            for (int x2 = 0; x2 < ow; ++x2)
            {
                const int sy = y / 2;
                const int sx = x2 / 2;
                const int sub = (y & 1) * 2 + (x2 & 1);
                for (int c = 0; c < 64; ++c)
                {
                    shuf[static_cast<size_t>(c) * ow * oh + y * ow + x2] =
                        up[static_cast<size_t>(c * 4 + sub) * plane +
                           static_cast<size_t>(sy) * w + sx];
                }
            }
        }
        std::vector<float> out(3 * static_cast<size_t>(ow) * oh);
        conv3x3(shuf.data(), 64, out.data(), 3, ow, oh, "module.tail.1",
                threads);
        for (size_t i = 0; i < static_cast<size_t>(ow) * oh; ++i)
        {
            out_r[i] = clamp01(out[i] / 255.0f);
            out_g[i] = clamp01(out[static_cast<size_t>(ow) * oh + i] / 255.0f);
            out_b[i] =
                clamp01(out[2 * static_cast<size_t>(ow) * oh + i] / 255.0f);
        }
    }

private:
    TorchStateDict st_;
    std::string error_;
    bool loaded_ = false;

    static float clamp01(float v)
    {
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }

    // The shared branch-free same-padding 3x3 (MLX/OIHW-agnostic: the
    // torch checkpoint is [out, in, kh, kw]).
    void conv3x3(const float* in, int in_ch, float* out, int out_ch, int w,
                 int h, const std::string& name, int threads) const
    {
        const Tensor& wt = st_.tensors.at(name + ".weight");
        const Tensor& bt = st_.tensors.at(name + ".bias");
        const float* wgt = wt.data.data();
        const float* bias = bt.data.data();
        const size_t plane = static_cast<size_t>(w) * h;
        const int nthreads = threads < 1 ? 1 : threads;
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t)
        {
            pool.emplace_back([&, t]() {
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
                        // [out, in, kh, kw] layout.
                        const float* kw = wgt +
                            (static_cast<size_t>(oc) * in_ch + ic) * 9;
                        for (int y = 0; y < h; ++y)
                        {
                            float* dp = op + y * w;
                            for (int ky = 0; ky < 3; ++ky)
                            {
                                // Zero padding (upstream conv3x3
                                // semantics): taps outside the image
                                // contribute 0, never a clamped sample.
                                const int yy = y + ky - 1;
                                if (yy < 0 || yy >= h)
                                {
                                    continue;
                                }
                                const float* row = ip +
                                    static_cast<size_t>(yy) * w;
                                const float k0 = kw[ky * 3 + 0];
                                const float k1 = kw[ky * 3 + 1];
                                const float k2 = kw[ky * 3 + 2];
                                for (int x = 1; x < w - 1; ++x)
                                {
                                    dp[x] += row[x - 1] * k0 + row[x] * k1 +
                                             row[x + 1] * k2;
                                }
                            }
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
                                                  + xx] * kw[ky * 3 + kx];
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
};

}  // namespace demo_edsr
