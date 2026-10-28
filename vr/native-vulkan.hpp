// SPDX-License-Identifier: MIT
// Copyright (c) 2026 the dlss5-linux contributors
//
// Minimal compute-capable Vulkan context for the native GPU stage: instance,
// a preferred-discrete physical device with a compute queue, logical device,
// command pool, device-local buffers with staging transfers, timestamp
// queries and one-shot dispatches. The CPU path in the family headers stays
// the deterministic reference; every GPU family is parity-tested against it.
//
// Shader sources are GLSL compiled by the test/tool runner with glslc (no
// compiled binaries are stored in the repository). There is no vendor
// extension requirement: plain Vulkan 1.1 compute with push constants.
#ifndef DLSS5_NR_NATIVE_VULKAN_HPP
#define DLSS5_NR_NATIVE_VULKAN_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "native-pre-post.hpp"
#include "native-vit.hpp"

#if defined(__linux__) && !defined(DLSS5_NR_NO_VULKAN)
#define DLSS5_NR_HAS_VULKAN 1
#include <vulkan/vulkan.h>
#else
#define DLSS5_NR_HAS_VULKAN 0
#endif

namespace dlss5_nr {

#if DLSS5_NR_HAS_VULKAN

// ---- the device -----------------------------------------------------------------------------------

struct VulkanDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    float timestamp_period = 0.0f;   // nanoseconds per timestamp tick
    bool has_timestamps = false;
    char name[256] = {};
};

inline void vulkan_close(VulkanDevice& d) {
    if (d.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(d.device);
        if (d.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(d.device, d.pool, nullptr);
        }
        vkDestroyDevice(d.device, nullptr);
    }
    if (d.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(d.instance, nullptr);
    }
    d = VulkanDevice{};
}

// Opens a compute-capable device. `opened` is set false when no usable
// Vulkan implementation exists (the callers record an environment gap and
// skip; that is not a kernel failure). `prefer_discrete` selects the
// discrete GPU when several devices qualify.
inline Status vulkan_open(VulkanDevice& out, bool& opened,
                          bool prefer_discrete = true) {
    opened = false;
    out = VulkanDevice{};
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dlss5-native";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &out.instance) != VK_SUCCESS) {
        out = VulkanDevice{};
        return Status::kOk;  // nothing created; `opened` stays false
    }
    std::uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(out.instance, &dev_count, nullptr);
    if (dev_count == 0) {
        vulkan_close(out);
        return Status::kOk;
    }
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(out.instance, &dev_count, devices.data());
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    std::uint32_t chosen_family = 0;
    float chosen_period = 0.0f;
    bool chosen_ts = false;
    int chosen_score = -1;
    for (std::uint32_t i = 0; i < dev_count; ++i) {
        std::uint32_t fam_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &fam_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> fams(fam_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &fam_count,
                                                 fams.data());
        for (std::uint32_t f = 0; f < fam_count; ++f) {
            if ((fams[f].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
                continue;
            }
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(devices[i], &props);
            // Prefer a timestamp-capable compute family on the same device.
            if (chosen == devices[i] && chosen_ts
                && fams[f].timestampValidBits == 0) {
                continue;
            }
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score = prefer_discrete ? 2 : 1;
            } else if (props.deviceType
                       == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score = prefer_discrete ? 1 : 2;
            }
            if (score > chosen_score) {
                chosen_score = score;
                chosen = devices[i];
                chosen_family = f;
                chosen_period = props.limits.timestampPeriod;
                chosen_ts = fams[f].timestampValidBits > 0;
                std::snprintf(out.name, sizeof(out.name), "%s",
                              props.deviceName);
            }
            break;  // one queue family per device suffices
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        vulkan_close(out);
        return Status::kOk;
    }
    out.physical = chosen;
    out.queue_family = chosen_family;
    out.timestamp_period = chosen_period;
    out.has_timestamps = chosen_ts;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = chosen_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(chosen, &dci, nullptr, &out.device) != VK_SUCCESS) {
        vulkan_close(out);
        return Status::kOk;
    }
    vkGetDeviceQueue(out.device, chosen_family, 0, &out.queue);
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = chosen_family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(out.device, &pci, nullptr, &out.pool)
        != VK_SUCCESS) {
        vulkan_close(out);
        return Status::kOk;
    }
    opened = true;
    return Status::kOk;
}

// ---- buffers --------------------------------------------------------------------------------------

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::size_t size = 0;
};

inline void vulkan_buffer_destroy(const VulkanDevice& d, VulkanBuffer& b) {
    if (b.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(d.device, b.buffer, nullptr);
    }
    if (b.memory != VK_NULL_HANDLE) {
        vkFreeMemory(d.device, b.memory, nullptr);
    }
    b = VulkanBuffer{};
}

inline Status vulkan_buffer_create(const VulkanDevice& d, VulkanBuffer& out,
                                   std::size_t bytes, VkBufferUsageFlags usage,
                                   bool host_visible) {
    out = VulkanBuffer{};
    if (bytes == 0) {
        return Status::kInvalidBuffer;
    }
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d.device, &bci, nullptr, &out.buffer) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d.device, out.buffer, &req);
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(d.physical, &props);
    const VkMemoryPropertyFlags want = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    std::uint32_t index = UINT32_MAX;
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) != 0
            && (props.memoryTypes[i].propertyFlags & want) == want) {
            index = i;
            break;
        }
    }
    if (index == UINT32_MAX) {
        vulkan_buffer_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = index;
    if (vkAllocateMemory(d.device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
        vulkan_buffer_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    if (vkBindBufferMemory(d.device, out.buffer, out.memory, 0)
        != VK_SUCCESS) {
        vulkan_buffer_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    out.size = bytes;
    return Status::kOk;
}

inline Status vulkan_upload(const VulkanDevice& d, VulkanBuffer& staging,
                            VulkanBuffer& device_local, const void* data,
                            std::size_t bytes) {
    if (data == nullptr || bytes > staging.size || bytes > device_local.size) {
        return Status::kInvalidBuffer;
    }
    void* mapped = nullptr;
    if (vkMapMemory(d.device, staging.memory, 0, bytes, 0, &mapped)
        != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(d.device, staging.memory);
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(cmd, staging.buffer, device_local.buffer, 1, &copy);
    ok = ok && vkEndCommandBuffer(cmd) == VK_SUCCESS;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ok = ok && vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
    ok = ok && vkQueueWaitIdle(d.queue) == VK_SUCCESS;
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    return ok ? Status::kOk : Status::kInvalidBuffer;
}

inline Status vulkan_download(const VulkanDevice& d, VulkanBuffer& staging,
                              VulkanBuffer& device_local, void* data,
                              std::size_t bytes) {
    if (data == nullptr || bytes > staging.size || bytes > device_local.size) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(cmd, device_local.buffer, staging.buffer, 1, &copy);
    ok = ok && vkEndCommandBuffer(cmd) == VK_SUCCESS;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ok = ok && vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
    ok = ok && vkQueueWaitIdle(d.queue) == VK_SUCCESS;
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    if (!ok) {
        return Status::kInvalidBuffer;
    }
    void* mapped = nullptr;
    if (vkMapMemory(d.device, staging.memory, 0, bytes, 0, &mapped)
        != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    std::memcpy(data, mapped, bytes);
    vkUnmapMemory(d.device, staging.memory);
    return Status::kOk;
}

// ---- shader modules --------------------------------------------------------------------------------

// Loads SPIR-V produced by the runner's glslc step (never stored in-tree).
inline Status vulkan_shader(const VulkanDevice& d, const std::uint32_t* words,
                            std::size_t word_count, VkShaderModule& out) {
    if (words == nullptr || word_count == 0) {
        return Status::kInvalidBuffer;
    }
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = word_count * sizeof(std::uint32_t);
    ci.pCode = words;
    if (vkCreateShaderModule(d.device, &ci, nullptr, &out) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// ---- the projection matmul (the hot family) --------------------------------------------------------

// out[t][n] = sum_k in[t][k] * w[n][k] on the GPU: the CPU's matmul_e4m3
// shape with e4m3 weights uploaded as exact float32 decodes. The shader's
// `precise` accumulation keeps the k-ordered scalar sum uncontracted and
// unreassociated for CPU bit-parity.
struct VulkanMatmul {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;
    VulkanBuffer in_dev, w_dev, out_dev, staging;
    int max_tokens = 0;
    int max_c_in = 0;
    int max_rows = 0;
};

struct VulkanMatmulTiming {
    double upload_ms = 0.0;
    double kernel_ms = 0.0;
    double download_ms = 0.0;
};

inline void vulkan_matmul_destroy(const VulkanDevice& d, VulkanMatmul& m) {
    if (d.device == VK_NULL_HANDLE) {
        m = VulkanMatmul{};
        return;
    }
    vkDeviceWaitIdle(d.device);
    vulkan_buffer_destroy(d, m.in_dev);
    vulkan_buffer_destroy(d, m.w_dev);
    vulkan_buffer_destroy(d, m.out_dev);
    vulkan_buffer_destroy(d, m.staging);
    if (m.queries != VK_NULL_HANDLE) {
        vkDestroyQueryPool(d.device, m.queries, nullptr);
    }
    if (m.dpool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(d.device, m.dpool, nullptr);
    }
    if (m.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(d.device, m.pipeline, nullptr);
    }
    if (m.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(d.device, m.layout, nullptr);
    }
    if (m.dsl != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(d.device, m.dsl, nullptr);
    }
    m = VulkanMatmul{};
}

inline Status vulkan_matmul_create(const VulkanDevice& d, VulkanMatmul& out,
                                   const std::uint32_t* spirv,
                                   std::size_t word_count, int max_tokens,
                                   int max_c_in, int max_rows) {
    if (spirv == nullptr || word_count == 0 || max_tokens < 1 || max_c_in < 1
        || max_rows < 1) {
        return Status::kInvalidValues;
    }
    vulkan_matmul_destroy(d, out);  // safe on a zeroed or live handle set
    out = VulkanMatmul{};
    out.max_tokens = max_tokens;
    out.max_c_in = max_c_in;
    out.max_rows = max_rows;
    VkShaderModule module = VK_NULL_HANDLE;
    {
        const Status st = vulkan_shader(d, spirv, word_count, module);
        if (st != Status::kOk) {
            return st;
        }
    }
    VkDescriptorSetLayoutBinding binds[3]{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 3 * sizeof(std::uint32_t);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &out.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    // Create layout and pipeline; on any failure clean up and report.
    bool ok = vkCreateDescriptorSetLayout(d.device, &dslci, nullptr,
                                          &out.dsl) == VK_SUCCESS;
    plci.pSetLayouts = &out.dsl;
    ok = ok && vkCreatePipelineLayout(d.device, &plci, nullptr, &out.layout)
                     == VK_SUCCESS;
    cpci.layout = out.layout;
    ok = ok
        && vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                    &out.pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(d.device, module, nullptr);
    VkDescriptorPoolSize psize{};
    psize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psize;
    ok = ok && vkCreateDescriptorPool(d.device, &dpci, nullptr, &out.dpool)
                     == VK_SUCCESS;
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = out.dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &out.dsl;
    ok = ok
        && vkAllocateDescriptorSets(d.device, &dsai, &out.set) == VK_SUCCESS;
    const std::size_t in_bytes = static_cast<std::size_t>(max_tokens)
        * static_cast<std::size_t>(max_c_in) * sizeof(float);
    const std::size_t w_bytes = static_cast<std::size_t>(max_rows)
        * static_cast<std::size_t>(max_c_in) * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(max_tokens)
        * static_cast<std::size_t>(max_rows) * sizeof(float);
    const std::size_t staging_bytes =
        in_bytes > w_bytes ? (in_bytes > out_bytes ? in_bytes : out_bytes)
                           : (w_bytes > out_bytes ? w_bytes : out_bytes);
    ok = ok
        && vulkan_buffer_create(d, out.in_dev, in_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                false)
               == Status::kOk;
    ok = ok
        && vulkan_buffer_create(d, out.w_dev, w_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                false)
               == Status::kOk;
    ok = ok
        && vulkan_buffer_create(d, out.out_dev, out_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                false)
               == Status::kOk;
    ok = ok
        && vulkan_buffer_create(d, out.staging, staging_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                true)
               == Status::kOk;
    VkDescriptorBufferInfo infos[3]{};
    infos[0] = {out.in_dev.buffer, 0, in_bytes};
    infos[1] = {out.w_dev.buffer, 0, w_bytes};
    infos[2] = {out.out_dev.buffer, 0, out_bytes};
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = out.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    if (ok) {
        vkUpdateDescriptorSets(d.device, 3, writes, 0, nullptr);
    }
    if (d.has_timestamps) {
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        ok = ok
            && vkCreateQueryPool(d.device, &qpci, nullptr, &out.queries)
                   == VK_SUCCESS;
    }
    if (!ok) {
        vulkan_matmul_destroy(d, out);
        return Status::kInvalidBuffer;
    }
    return Status::kOk;
}

// Runs out[t][n] = sum_k in[t][k] * w[n][k] with the CPU's exact element
// order; `in`/`weights` are plain row-major float arrays (the weights are
// the CPU's exact e4m3 decodes).
inline Status vulkan_matmul_run(const VulkanDevice& d, VulkanMatmul& m,
                                const float* in, int tokens, int c_in,
                                const float* weights, int rows, float* out,
                                VulkanMatmulTiming& timing) {
    if (in == nullptr || weights == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || c_in < 1 || rows < 1 || tokens > m.max_tokens
        || c_in > m.max_c_in || rows > m.max_rows) {
        return Status::kInvalidDimensions;
    }
    // The 2D dispatch count must stay within the guaranteed workgroup
    // count limits and the shader's uint32 indexing.
    if (static_cast<std::uint64_t>(tokens) > 65535u
        || static_cast<std::uint64_t>(rows) > (65535u * 128u)
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(rows)
               > UINT32_MAX) {
        return Status::kInvalidDimensions;
    }
    timing = VulkanMatmulTiming{};
    const std::size_t in_bytes = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(c_in) * sizeof(float);
    const std::size_t w_bytes = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(c_in) * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(rows) * sizeof(float);
    double ms = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        Status st = vulkan_upload(d, m.staging, m.in_dev, in, in_bytes);
        if (st == Status::kOk) {
            st = vulkan_upload(d, m.staging, m.w_dev, weights, w_bytes);
        }
        const auto t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing.upload_ms = ms;
        if (st != Status::kOk) {
            return st;
        }
    }
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cai, &cmd) != VK_SUCCESS) {
        return Status::kInvalidBuffer;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return Status::kInvalidBuffer;
    }
    if (m.queries != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m.queries, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m.queries,
                            0);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m.layout, 0,
                            1, &m.set, 0, nullptr);
    const std::uint32_t push[3] = {static_cast<std::uint32_t>(tokens),
                                   static_cast<std::uint32_t>(c_in),
                                   static_cast<std::uint32_t>(rows)};
    vkCmdPushConstants(cmd, m.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), push);
    // 2D dispatch: x = output rows (128-wide), y = tokens.
    vkCmdDispatch(cmd, (push[2] + 127) / 128, push[0], 1);
    if (m.queries != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            m.queries, 1);
    }
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return Status::kInvalidBuffer;
    }
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    bool submitted = false;
    {
        const auto t0 = std::chrono::steady_clock::now();
        submitted = vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE)
                    == VK_SUCCESS;
        submitted = (vkQueueWaitIdle(d.queue) == VK_SUCCESS) && submitted;
        const auto t1 = std::chrono::steady_clock::now();
        timing.kernel_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    if (!submitted) {
        return Status::kInvalidBuffer;
    }
    if (m.queries != VK_NULL_HANDLE && d.has_timestamps) {
        std::uint64_t stamps[2] = {0, 0};
        if (vkGetQueryPoolResults(d.device, m.queries, 0, 2, sizeof(stamps),
                                  stamps, sizeof(stamps[0]),
                                  VK_QUERY_RESULT_64_BIT)
            == VK_SUCCESS) {
            const double ticks =
                static_cast<double>(stamps[1] - stamps[0]);
            timing.kernel_ms = ticks * static_cast<double>(d.timestamp_period)
                / 1.0e6;
        }
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        const Status st = vulkan_download(d, m.staging, m.out_dev, out,
                                          out_bytes);
        const auto t1 = std::chrono::steady_clock::now();
        timing.download_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (st != Status::kOk) {
            return st;
        }
    }
    return Status::kOk;
}

// ---- the projection executor (the G2 eval-dispatch backend) ---------------------------------------

// Binds a VulkanMatmul as a ProjectionExecutor for the eval dispatch's vit
// projection rows. The e4m3 codes are decoded host-side (exact in float32)
// per call; G4 weight residency replaces this with persistent buffers.
struct VulkanProjectionSite {
    VulkanDevice* device = nullptr;
    VulkanMatmul matmul;         // grow-only: resized to cover row shapes
    std::uint32_t* spirv = nullptr;
    std::size_t spirv_words = 0;
    std::vector<float> decoded;
    double decode_ms = 0.0;
    double kernel_ms = 0.0;
    double upload_ms = 0.0;
    double download_ms = 0.0;
};

inline void vulkan_projection_close(VulkanProjectionSite& site) {
    if (site.device != nullptr) {
        vulkan_matmul_destroy(*site.device, site.matmul);
    }
    site = VulkanProjectionSite{};
}

inline Status vulkan_projection_matmul(void* ctx, const float* in, int tokens,
                                       int c_in, const E4m3Matrix& w,
                                       float* out) {
    auto* site = static_cast<VulkanProjectionSite*>(ctx);
    if (site == nullptr || site->device == nullptr || site->spirv == nullptr) {
        return Status::kNullPointer;
    }
    if (in == nullptr || w.bits == nullptr || out == nullptr) {
        return Status::kNullPointer;
    }
    if (tokens < 1 || c_in < 1 || w.rows < 1 || w.cols != c_in) {
        return Status::kInvalidDimensions;
    }
    if (tokens > 65535 || w.rows > 65535 * 128
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(w.rows)
               > UINT32_MAX
        || static_cast<std::uint64_t>(tokens)
                   * static_cast<std::uint64_t>(c_in)
               > UINT32_MAX
        || static_cast<std::uint64_t>(w.rows)
                   * static_cast<std::uint64_t>(c_in)
               > UINT32_MAX) {
        return Status::kInvalidDimensions;
    }
    // Grow-only coverage: the plan's row shapes stabilise after a few
    // resizes; the buffers never shrink.
    if (site->matmul.pipeline == VK_NULL_HANDLE
        || tokens > site->matmul.max_tokens || c_in > site->matmul.max_c_in
        || w.rows > site->matmul.max_rows) {
        // Grow-only: never shrink, so alternating row shapes (expand
        // 1024->4096 / contract 4096->1024) settle after at most one
        // rebuild per dimension and stop rebuilding. The old maxima are
        // read before the destroy zeroes them.
        int max_tokens = site->matmul.max_tokens > tokens
            ? site->matmul.max_tokens : tokens;
        int max_c_in = site->matmul.max_c_in > c_in ? site->matmul.max_c_in
                                                    : c_in;
        int max_rows = site->matmul.max_rows > w.rows ? site->matmul.max_rows
                                                      : w.rows;
        if (max_tokens < 4096) { max_tokens = 4096; }
        if (max_c_in < 1024) { max_c_in = 1024; }
        if (max_rows < 1024) { max_rows = 1024; }
        vulkan_matmul_destroy(*site->device, site->matmul);
        const Status st = vulkan_matmul_create(*site->device, site->matmul,
                                               site->spirv, site->spirv_words,
                                               max_tokens, max_c_in, max_rows);
        if (st != Status::kOk) {
            return st;
        }
    }
    const std::size_t count = static_cast<std::size_t>(w.rows)
        * static_cast<std::size_t>(c_in);
    const auto d0 = std::chrono::steady_clock::now();
    site->decoded.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        site->decoded[i] = e4m3_from_bits(w.bits[i]);
    }
    const auto d1 = std::chrono::steady_clock::now();
    site->decode_ms +=
        std::chrono::duration<double, std::milli>(d1 - d0).count();
    VulkanMatmulTiming timing;
    const Status st = vulkan_matmul_run(*site->device, site->matmul, in,
                                        tokens, c_in, site->decoded.data(),
                                        w.rows, out, timing);
    site->kernel_ms += timing.kernel_ms;
    site->upload_ms += timing.upload_ms;
    site->download_ms += timing.download_ms;
    return st;
}

inline ProjectionExecutor vulkan_projection_executor(VulkanProjectionSite*
                                                         site) {
    ProjectionExecutor ex;
    ex.ctx = site;
    ex.matmul = &vulkan_projection_matmul;
    return ex;
}

#endif  // DLSS5_NR_HAS_VULKAN

}  // namespace dlss5_nr

#endif  // DLSS5_NR_NATIVE_VULKAN_HPP
