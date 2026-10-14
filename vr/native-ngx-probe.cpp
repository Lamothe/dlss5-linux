// Native Linux capability/creation probe. No Wine, rendering, inference or model extraction.
#include <vulkan/vulkan.h>
#include <nvsdk_ngx_vk.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
static_assert(NVSDK_NGX_Result_Success == 0x1);
static_assert(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature == 0xbad0000b);
static_assert(NVSDK_NGX_Result_FAIL_OutOfDate == 0xbad0000c);
static_assert(NVSDK_NGX_Result_FAIL_FeatureNotFound == 0xbad00004);
static_assert(NVSDK_NGX_Result_FAIL_UnsupportedParameter == 0xbad00010);
constexpr const char* project_id = "a642c19a-4503-4edc-897d-2198a2ecf702";
void vk_check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " VkResult=" + std::to_string(result));
    }
}
void ngx_check(NVSDK_NGX_Result result, const char* operation) {
    std::printf("%s result=0x%08x\n", operation, static_cast<unsigned int>(result));
    if (result != NVSDK_NGX_Result_Success) {
        throw std::runtime_error(operation);
    }
}
void log_callback(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature feature) {
    std::fprintf(stderr, "NGX feature=%d %s\n", static_cast<int>(feature), message);
}
struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    NVSDK_NGX_Parameter* capabilities = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* handle = nullptr;
    bool initialised = false;
    bool gpu_pending = false;
    void cleanup() {
        // Never destroy resources underneath unfinished GPU work or wait without a bound.
        if (gpu_pending && vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ULL) != VK_SUCCESS) {
            std::fprintf(stderr, "GPU completion failed during cleanup\n");
            std::fflush(nullptr);
            std::_Exit(1);
        }
        gpu_pending = false;
        if (handle) {
            ngx_check(NVSDK_NGX_VULKAN_ReleaseFeature(handle), "Cleanup ReleaseFeature");
            handle = nullptr;
        }
        if (parameters) {
            ngx_check(NVSDK_NGX_VULKAN_DestroyParameters(parameters), "DestroyParameters");
            parameters = nullptr;
        }
        if (capabilities) {
            ngx_check(NVSDK_NGX_VULKAN_DestroyParameters(capabilities), "DestroyCapabilityParameters");
            capabilities = nullptr;
        }
        if (initialised) {
            ngx_check(NVSDK_NGX_VULKAN_Shutdown1(device), "Shutdown");
            initialised = false;
        }
        if (fence) { vkDestroyFence(device, fence, nullptr); }
        if (pool) { vkDestroyCommandPool(device, pool, nullptr); }
        if (device) { vkDestroyDevice(device, nullptr); }
        if (instance) { vkDestroyInstance(instance, nullptr); }
        fence = VK_NULL_HANDLE; pool = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE; instance = VK_NULL_HANDLE;
    }
    ~Context() {
        try { cleanup(); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "REJECT cleanup: %s\n", error.what());
            std::fflush(nullptr);
            std::_Exit(1);
        }
    }
    NVSDK_NGX_Result create(NVSDK_NGX_Feature feature) {
        vk_check(vkResetCommandBuffer(command, 0), "reset command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(command, &begin), "begin command buffer");
        const auto result = NVSDK_NGX_VULKAN_CreateFeature1(device, command, feature, parameters, &handle);
        std::printf("Create feature=%d result=0x%08x handle=%s\n", static_cast<int>(feature),
                    static_cast<unsigned int>(result), handle ? "nonnull" : "null");
        vk_check(vkEndCommandBuffer(command), "end command buffer");
        // Creation can record GPU setup commands, even when its reported result fails.
        vk_check(vkResetFences(device, 1, &fence), "reset fence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        vk_check(vkQueueSubmit(queue, 1, &submit, fence), "queue submit");
        gpu_pending = true;
        vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ULL), "wait fence");
        gpu_pending = false;
        if ((result == NVSDK_NGX_Result_Success) != (handle != nullptr)) {
            throw std::runtime_error("create result/handle mismatch");
        }
        if (handle) {
            const auto release = NVSDK_NGX_VULKAN_ReleaseFeature(handle);
            ngx_check(release, "ReleaseFeature");
            handle = nullptr;
        }
        return result;
    }
};
void check_capability(NVSDK_NGX_Parameter* parameters, const char* name, NVSDK_NGX_Result expected_result, int expected_value) {
    int value = -12345;
    const auto result = parameters->Get(name, &value);
    std::printf("Capability %s result=0x%08x value=%d key=%s\n", name, static_cast<unsigned int>(result), value,
                result == NVSDK_NGX_Result_Success ? "present" : "missing/unsupported");
    if (result != expected_result || value != expected_value) {
        throw std::runtime_error(std::string("unexpected capability result/value: ") + name);
    }
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 3 || (std::strcmp(argv[2], "with-sr") != 0 && std::strcmp(argv[2], "without-sr") != 0)) {
        std::fprintf(stderr, "Usage: native-ngx-probe SDK_LIB_DIR with-sr|without-sr\n");
        return 1;
    }
    try {
        Context context;
        const bool with_sr = std::strcmp(argv[2], "with-sr") == 0;
        const std::wstring search_path = std::filesystem::absolute(argv[1]).wstring();
        const wchar_t* paths[] = {search_path.c_str()};
        NVSDK_NGX_FeatureCommonInfo info{};
        info.PathListInfo.Path = paths;
        info.PathListInfo.Length = 1;
        info.LoggingInfo.LoggingCallback = log_callback;
        info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
        info.LoggingInfo.DisableOtherLoggingSinks = true;
        unsigned int instance_count = 0, device_count = 0;
        const char** instance_extensions = nullptr;
        const char** required_device_extensions = nullptr;
        ngx_check(NVSDK_NGX_VULKAN_RequiredExtensions(&instance_count, &instance_extensions,
                  &device_count, &required_device_extensions), "RequiredExtensions");
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "Native Linux NGX feasibility probe";
        application.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &application;
        instance_info.enabledExtensionCount = instance_count;
        instance_info.ppEnabledExtensionNames = instance_extensions;
        vk_check(vkCreateInstance(&instance_info, nullptr, &context.instance), "create instance");
        uint32_t physical_count = 0;
        vk_check(vkEnumeratePhysicalDevices(context.instance, &physical_count, nullptr), "count devices");
        std::vector<VkPhysicalDevice> physical_devices(physical_count);
        vk_check(vkEnumeratePhysicalDevices(context.instance, &physical_count, physical_devices.data()), "enumerate devices");
        for (auto physical : physical_devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical, &properties);
            std::printf("Device vendor=0x%x device=0x%x name=%s\n", properties.vendorID, properties.deviceID, properties.deviceName);
            if (properties.vendorID == 0x10de && std::strcmp(properties.deviceName, "NVIDIA GeForce RTX 4070 Ti SUPER") == 0) {
                context.physical = physical;
            }
        }
        if (!context.physical) { throw std::runtime_error("expected RTX 4070 Ti SUPER not present"); }
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(context.physical, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(context.physical, &queue_count, queues.data());
        uint32_t family = queue_count;
        for (uint32_t i = 0; i < queue_count; ++i) {
            const auto required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if ((queues[i].queueFlags & required) == required && queues[i].queueCount != 0) { family = i; break; }
        }
        if (family == queue_count) { throw std::runtime_error("no graphics/compute queue"); }
        uint32_t extension_count = 0;
        vk_check(vkEnumerateDeviceExtensionProperties(context.physical, nullptr, &extension_count, nullptr), "count extensions");
        std::vector<VkExtensionProperties> available(extension_count);
        vk_check(vkEnumerateDeviceExtensionProperties(context.physical, nullptr, &extension_count, available.data()), "enumerate extensions");
        std::vector<const char*> extensions;
        for (unsigned int i = 0; i < device_count; ++i) {
            const char* name = required_device_extensions[i];
            const bool found = std::any_of(available.begin(), available.end(), [name](const auto& p) { return std::strcmp(name, p.extensionName) == 0; });
            if (!found) { throw std::runtime_error(std::string("missing required extension ") + name); }
            std::printf("DeviceExtension %s\n", name);
            extensions.push_back(name);
        }
        // NGX explicitly requires EXT BDA. Do not mix this with core/KHR BDA
        // or enable every advertised Vulkan 1.1/1.2 feature (VUID-04748).
        const bool ext_bda = std::any_of(extensions.begin(), extensions.end(), [](const char* name) {
            return std::strcmp(name, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0;
        });
        if (!ext_bda) { throw std::runtime_error("pinned SDK no longer requires EXT buffer device address"); }
        for (const char* name : extensions) {
            if (std::strcmp(name, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
                throw std::runtime_error("cannot enable both EXT and KHR buffer device address");
            }
        }
        VkPhysicalDeviceBufferDeviceAddressFeaturesEXT supported_bda{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        supported.pNext = &supported_bda;
        vkGetPhysicalDeviceFeatures2(context.physical, &supported);
        if (!supported_bda.bufferDeviceAddress) { throw std::runtime_error("EXT buffer device address unsupported"); }
        VkPhysicalDeviceBufferDeviceAddressFeaturesEXT enabled_bda{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT};
        enabled_bda.bufferDeviceAddress = VK_TRUE;
        std::printf("DeviceFeature EXT.bufferDeviceAddress=1; all other optional features disabled\n");
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family; queue_info.queueCount = 1; queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.pNext = &enabled_bda;
        device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        device_info.ppEnabledExtensionNames = extensions.data();
        vk_check(vkCreateDevice(context.physical, &device_info, nullptr, &context.device), "create device");
        vkGetDeviceQueue(context.device, family, 0, &context.queue);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pool_info.queueFamilyIndex = family;
        vk_check(vkCreateCommandPool(context.device, &pool_info, nullptr, &context.pool), "create command pool");
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = context.pool; command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; command_info.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(context.device, &command_info, &context.command), "allocate command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vk_check(vkCreateFence(context.device, &fence_info, nullptr, &context.fence), "create fence");
        ngx_check(NVSDK_NGX_VULKAN_Init_with_ProjectID(project_id, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", L".",
                  context.instance, context.physical, context.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &info), "Init");
        context.initialised = true;
        ngx_check(NVSDK_NGX_VULKAN_GetCapabilityParameters(&context.capabilities), "GetCapabilityParameters");
        if (!context.capabilities) { throw std::runtime_error("null capabilities"); }
        // Expectations from the saved with/without-SR runs. Availability is
        // distinct from an absent parameter key; FeatureInitResult uses SDK codes.
        const auto unavailable_init = static_cast<int>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
        check_capability(context.capabilities, "SuperSampling.Available", NVSDK_NGX_Result_Success, with_sr ? 1 : 0);
        check_capability(context.capabilities, "SuperSampling.FeatureInitResult", NVSDK_NGX_Result_Success,
                         with_sr ? static_cast<int>(NVSDK_NGX_Result_Success) : unavailable_init);
        check_capability(context.capabilities, "DLSSNR.Available", NVSDK_NGX_Result_Success, 0);
        check_capability(context.capabilities, "DLSSNR.FeatureInitResult", NVSDK_NGX_Result_Success, unavailable_init);
        check_capability(context.capabilities, "DLSSNR.NeedsUpdatedDriver", NVSDK_NGX_Result_Success, 0);
        check_capability(context.capabilities, "Reserved18.Available", NVSDK_NGX_Result_FAIL_UnsupportedParameter, -12345);
        ngx_check(NVSDK_NGX_VULKAN_AllocateParameters(&context.parameters), "AllocateParameters");
        if (!context.parameters) { throw std::runtime_error("null parameters"); }
        context.parameters->Set(NVSDK_NGX_Parameter_Width, 720u);
        context.parameters->Set(NVSDK_NGX_Parameter_Height, 720u);
        context.parameters->Set(NVSDK_NGX_Parameter_OutWidth, 720u);
        context.parameters->Set(NVSDK_NGX_Parameter_OutHeight, 720u);
        context.parameters->Set(NVSDK_NGX_Parameter_PerfQualityValue, static_cast<int>(NVSDK_NGX_PerfQuality_Value_DLAA));
        context.parameters->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, 0);
        context.parameters->Set("DLSSNR.Enabled", 1u);
        context.parameters->Set("DLSSNR.Width", 720u);
        context.parameters->Set("DLSSNR.Height", 720u);
        context.parameters->Set("DLSSNR.Hint.Render.Preset", 0u);
        context.parameters->Set("DLSSNR.UICorrection", 1u);
        context.parameters->Set("CreationNodeMask", 1u);
        context.parameters->Set("VisibilityNodeMask", 1u);
        const auto sr_result = context.create(NVSDK_NGX_Feature_SuperSampling);
        // Feature 18 is NR in the tested Windows runtime; public Linux headers still reserve it.
        const auto nr_result = context.create(NVSDK_NGX_Feature_Reserved18);
        const auto invalid_result = context.create(static_cast<NVSDK_NGX_Feature>(12345));
        // These are the actual pinned SDK names: 0xbad0000b is NOT
        // FeatureNotSupported, and 0xbad0000c is NOT FeatureNotFound.
        std::printf("Expected rejection codes: UnableToInitializeFeature=0x%08x OutOfDate=0x%08x\n",
                    static_cast<unsigned int>(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature),
                    static_cast<unsigned int>(NVSDK_NGX_Result_FAIL_OutOfDate));
        const auto expected_sr = with_sr ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
        if (sr_result != expected_sr || nr_result != NVSDK_NGX_Result_FAIL_UnableToInitializeFeature ||
            invalid_result != NVSDK_NGX_Result_FAIL_OutOfDate) {
            throw std::runtime_error("exact saved-log control result unexpected");
        }
        context.cleanup();
        std::printf("CONTROL PASS sr_present=%d sr_created=%d invalid_rejected=1 nr_created=0; cleanup successful; NO evaluations performed\n",
                     with_sr, sr_result == NVSDK_NGX_Result_Success);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "REJECT: %s\n", error.what());
        return 1;
    }
}
