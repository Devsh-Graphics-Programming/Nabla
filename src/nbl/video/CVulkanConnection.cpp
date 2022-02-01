#include "nbl/video/CVulkanConnection.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/debug/CVulkanDebugCallback.h"

#define LOG(logger, ...) if (logger) {logger->log(__VA_ARGS__);}

namespace nbl::video
{
    core::smart_refctd_ptr<CVulkanConnection> CVulkanConnection::create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        const uint32_t requiredFeatureCount, video::IAPIConnection::E_FEATURE* requiredFeatures,
        const uint32_t optionalFeatureCount, video::IAPIConnection::E_FEATURE* optionalFeatures,
        core::smart_refctd_ptr<system::ILogger>&& logger, bool enableValidation)
    {
        if (volkInitialize() != VK_SUCCESS)
        {
            LOG(logger, "Failed to initialize volk!\n", system::ILogger::ELL_ERROR);
            return nullptr;
        }

        auto getAvailableLayers = [](uint32_t& layerCount, VkLayerProperties* layers) -> bool
        {
            uint32_t count;
            VkResult retval = vkEnumerateInstanceLayerProperties(&count, nullptr);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
                return false;

            retval = vkEnumerateInstanceLayerProperties(&count, layers);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
                return false;

            layerCount += count;

            return true;
        };

        constexpr uint32_t MAX_LAYER_COUNT = 100u;
        constexpr uint32_t MAX_EXTENSION_COUNT = (1u << 12) / sizeof(char*);

        const size_t memSizeNeeded = MAX_EXTENSION_COUNT * sizeof(VkExtensionProperties) + MAX_LAYER_COUNT * sizeof(VkLayerProperties);
        void* mem = _NBL_ALIGNED_MALLOC(memSizeNeeded, _NBL_SIMD_ALIGNMENT);
        auto memFree = core::makeRAIIExiter([mem] {_NBL_ALIGNED_FREE(mem); });

        VkExtensionProperties* availableExtensions = static_cast<VkExtensionProperties*>(mem);
        VkLayerProperties* availableLayers = reinterpret_cast<VkLayerProperties*>(availableExtensions + MAX_EXTENSION_COUNT);

        // Get available layers
        uint32_t availableLayerCount = 0u;
        if (!getAvailableLayers(availableLayerCount, availableLayers))
            return nullptr;
        assert(availableLayerCount <= MAX_LAYER_COUNT);

        const char* requiredLayerNames[MAX_LAYER_COUNT] = { nullptr };
        uint32_t requiredLayerNameCount = 0u;
        {
            if (enableValidation)
                requiredLayerNames[requiredLayerNameCount++] = "VK_LAYER_KHRONOS_validation";
        }
        assert(requiredLayerNameCount <= MAX_LAYER_COUNT);

        const bool layersSupported = std::all_of(requiredLayerNames, requiredLayerNames + requiredLayerNameCount,
            [availableLayers, availableLayerCount, &logger](const char* layerName)
            {
                const VkLayerProperties* retval = std::find_if(availableLayers, availableLayers + availableLayerCount,
                    [layerName](const VkLayerProperties& layerProps)
                    {
                        return strcmp(layerName, layerProps.layerName) == 0;
                    });

                if (retval == (availableLayers + availableLayerCount))
                {
                    LOG(logger, "Failed to find required instance layer: %s\n", system::ILogger::ELL_ERROR, layerName);
                    return false;
                }

                return true;
            });

        if (!layersSupported)
            return nullptr;

        using FeatureSetType = core::unordered_set<core::string>;

        auto getAvailableFeatureSet = [&logger, requiredLayerNameCount, requiredLayerNames](VkExtensionProperties* extensions) -> FeatureSetType
        {
            uint32_t totalCount = 0u;
            uint32_t count;

            if (getExtensionsForLayer(nullptr, count, extensions))
                totalCount += count;
            else
                LOG(logger, "Failed to get implicit instance extensions!\n");

            for (uint32_t i = 0u; i < requiredLayerNameCount; ++i)
            {
                if (getExtensionsForLayer(requiredLayerNames[i], count, extensions + totalCount))
                    totalCount += count;
                else
                    LOG(logger, "Failed to get instance extensions for the layer: %s\n", system::ILogger::ELL_ERROR, requiredLayerNames[i]);
            }

            FeatureSetType result;
            for (uint32_t i = 0; i < totalCount; ++i)
                result.insert(extensions[i].extensionName);

            return result;
        };

        FeatureSetType availableFeatureSet = getAvailableFeatureSet(availableExtensions);

        auto insertFeatureIfAvailable = [&logger, &availableFeatureSet](const E_FEATURE feature, auto& featureSet) -> bool
        {
            const auto depFeatures = IAPIConnection::getDependentFeatures(feature);

            constexpr uint32_t MAX_VULKAN_NAME_COUNT_PER_FEATURE = 8u;  // each feature/extension can spawn multiple extension names (mostly platform-specific ones)
            constexpr uint32_t MAX_FEATURE_COUNT = 1u << 9;
            const char* vulkanNames[MAX_FEATURE_COUNT * MAX_VULKAN_NAME_COUNT_PER_FEATURE];
            uint32_t vulkanNameCount = 0u;
            for (const auto& depFeature : depFeatures)
            {
                uint32_t count = 0u;
                getVulkanExtensionNamesFromFeature(depFeature, count, vulkanNames + vulkanNameCount);

                for (uint32_t index = 0u; index < count; ++index)
                {
                    if (availableFeatureSet.find(vulkanNames[vulkanNameCount + index]) == availableFeatureSet.end())
                    {
                        LOG(logger, "Failed to find instance extension: %s\n", system::ILogger::ELL_ERROR, vulkanNames[vulkanNameCount + index]);
                        return false;
                    }
                }
                vulkanNameCount += count;
            }
            assert(vulkanNameCount <= MAX_FEATURE_COUNT * MAX_VULKAN_NAME_COUNT_PER_FEATURE);

            featureSet.insert(vulkanNames, vulkanNames + vulkanNameCount);

            return true;
        };

        FeatureSetType selectedFeatureSet;
        for (uint32_t i = 0u; i < requiredFeatureCount; ++i)
        {
            if (!insertFeatureIfAvailable(requiredFeatures[i], selectedFeatureSet))
                return nullptr;
        }

        for (uint32_t i = 0u; i < optionalFeatureCount; ++i)
        {
            if (!insertFeatureIfAvailable(optionalFeatures[i], selectedFeatureSet))
                continue;
        }

        const size_t totalFeatureCount = selectedFeatureSet.size() + 1ull;
        core::vector<const char*> selectedFeatures(totalFeatureCount);
        uint32_t k = 0u;
        for (const auto& feature : selectedFeatureSet)
            selectedFeatures[k++] = feature.c_str();
        selectedFeatures[k++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

        std::unique_ptr<CVulkanDebugCallback> debugCallback = nullptr;
        VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        if (logger)
        {
            auto logLevelMask = logger->getLogLevelMask();
            debugCallback = std::make_unique<CVulkanDebugCallback>(std::move(logger));

            debugMessengerCreateInfo.pNext = nullptr;
            debugMessengerCreateInfo.flags = 0;
            auto debugCallbackFlags = getDebugCallbackFlagsFromLogLevelMask(logLevelMask);
            debugMessengerCreateInfo.messageSeverity = debugCallbackFlags.first;
            debugMessengerCreateInfo.messageType = debugCallbackFlags.second;
            debugMessengerCreateInfo.pfnUserCallback = CVulkanDebugCallback::defaultCallback;
            debugMessengerCreateInfo.pUserData = debugCallback.get();
        }
        
        uint32_t instanceApiVersion = MinimumVulkanApiVersion;
        vkEnumerateInstanceVersion(&instanceApiVersion); // Get Highest
        if(instanceApiVersion < MinimumVulkanApiVersion)
        {
            assert(false);
            return nullptr;
        }

        VkInstance vk_instance;
        {
            VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
            applicationInfo.pNext = nullptr; // pNext must be NULL
            applicationInfo.pApplicationName = appName;
            applicationInfo.applicationVersion = appVer;
            applicationInfo.pEngineName = "Nabla";
            applicationInfo.apiVersion = instanceApiVersion;
            applicationInfo.engineVersion = NABLA_VERSION_INTEGER;

            VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
            createInfo.pNext = debugCallback ? (VkDebugUtilsMessengerCreateInfoEXT*)&debugMessengerCreateInfo : nullptr;
            createInfo.flags = static_cast<VkInstanceCreateFlags>(0);
            createInfo.pApplicationInfo = &applicationInfo;
            createInfo.enabledLayerCount = requiredLayerNameCount;
            createInfo.ppEnabledLayerNames = requiredLayerNames;
            createInfo.enabledExtensionCount = static_cast<uint32_t>(selectedFeatures.size());
            createInfo.ppEnabledExtensionNames = selectedFeatures.data();

            if (vkCreateInstance(&createInfo, nullptr, &vk_instance) != VK_SUCCESS)
                return nullptr;
        }

        volkLoadInstanceOnly(vk_instance);

        constexpr uint32_t MAX_PHYSICAL_DEVICE_COUNT = 16u;
        uint32_t physicalDeviceCount = 0u;
        VkPhysicalDevice vk_physicalDevices[MAX_PHYSICAL_DEVICE_COUNT];
        {
            VkResult retval = vkEnumeratePhysicalDevices(vk_instance, &physicalDeviceCount, nullptr);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            {
                if (debugCallback)
                    LOG(debugCallback->getLogger(), "Failed to enumerate physical devices!\n");
                return nullptr;
            }

            if (physicalDeviceCount > MAX_PHYSICAL_DEVICE_COUNT)
            {
                if (debugCallback)
                    LOG(debugCallback->getLogger(), "Too many physical devices (%d) found!", system::ILogger::ELL_ERROR, physicalDeviceCount);
                return nullptr;
            }

            vkEnumeratePhysicalDevices(vk_instance, &physicalDeviceCount, vk_physicalDevices);
        }

        VkDebugUtilsMessengerEXT vk_debugMessenger = VK_NULL_HANDLE;
        if (debugCallback)
        {
            if (vkCreateDebugUtilsMessengerEXT(vk_instance, &debugMessengerCreateInfo, nullptr, &vk_debugMessenger) != VK_SUCCESS)
                return nullptr;
        }

        CVulkanConnection* apiRaw = new CVulkanConnection(vk_instance, std::move(debugCallback), vk_debugMessenger);
        core::smart_refctd_ptr<CVulkanConnection> api(apiRaw, core::dont_grab);
        auto& physicalDevices = api->m_physicalDevices;
        physicalDevices.reserve(physicalDeviceCount);
        for (uint32_t i = 0u; i < physicalDeviceCount; ++i)
        {
            physicalDevices.emplace_back(std::make_unique<CVulkanPhysicalDevice>(
                core::smart_refctd_ptr(sys),
                core::make_smart_refctd_ptr<asset::IGLSLCompiler>(sys.get()),
                api.get(), api->m_rdoc_api, vk_physicalDevices[i], vk_instance, instanceApiVersion));

        }

        return api;
    }

    CVulkanConnection::CVulkanConnection(
        VkInstance instance,
        std::unique_ptr<CVulkanDebugCallback>&& debugCallback,
        VkDebugUtilsMessengerEXT vk_debugMessenger)
        : IAPIConnection(), m_vkInstance(instance), m_debugCallback(std::move(debugCallback)),
        m_vkDebugUtilsMessengerEXT(vk_debugMessenger)
    {}

    CVulkanConnection::~CVulkanConnection()
    {
        if (m_vkDebugUtilsMessengerEXT != VK_NULL_HANDLE)
            vkDestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugUtilsMessengerEXT, nullptr);

        vkDestroyInstance(m_vkInstance, nullptr);
    }

    IDebugCallback* CVulkanConnection::getDebugCallback() const { return m_debugCallback.get(); }

    void CVulkanConnection::getVulkanExtensionNamesFromFeature(const IAPIConnection::E_FEATURE feature, uint32_t& extNameCount, const char** extNames)
    {
        extNameCount = 0u;

        switch (feature)
        {
        case IAPIConnection::EF_SURFACE:
        {
            extNames[extNameCount++] = VK_KHR_SURFACE_EXTENSION_NAME;
#if defined(_NBL_PLATFORM_WINDOWS_)
            extNames[extNameCount++] = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
#endif
        } break;

        default:
            break;
        }

        assert(extNameCount <= 8u); // it is rare that any feature will spawn more than 8 "variations" (usually due to OS-specific stuff), consequently the caller might only provide enough memory to write <= 8 of them
    }
}