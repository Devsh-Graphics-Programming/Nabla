#include "nbl/video/CVulkanConnection.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/debug/CVulkanDebugCallback.h"

#define LOG(logger, ...) if (logger) {logger->log(__VA_ARGS__);}

namespace nbl::video
{
    core::smart_refctd_ptr<CVulkanConnection> CVulkanConnection::create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        core::smart_refctd_ptr<system::ILogger>&& logger, const SFeatures& featuresToEnable)
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
            if (featuresToEnable.validations)
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
        
        FeatureSetType selectedFeatureSet;
        bool allRequestedFeaturesSupported = true;

        auto insertToFeatureSetIfAvailable = [&](const char* extStr, const char* featureName)
        {
            bool found = availableFeatureSet.find(extStr) != availableFeatureSet.end();
            
            if (found)
            {
                selectedFeatureSet.insert(extStr);
            }
            else
            {
                LOG(logger, "Feature Unavailable: %s\n", system::ILogger::ELL_ERROR, featureName);
                allRequestedFeaturesSupported = false;
            }
        };
        auto patchDependencies = [](FeatureSetType& selectedFeatureSet, SFeatures& actualFeaturesToEnable) -> void
        {
            // Vulkan Spec:
            // If an extension is supported (as queried by vkEnumerateInstanceExtensionProperties or
            //    vkEnumerateDeviceExtensionProperties), then required extensions of that extension must also be
            //    supported for the same instance or physical device.
            // -> So No need to use `insertToFeatureSetIfAvailable` because when vulkan reports an extension as supported it also has their dependancies supported
            // TODO: No current extension needs another, except when we add DISPLAY Swapchain mode because:
            // VK_KHR_display Requires VK_KHR_surface to be enabled
            return;
        };


        if(featuresToEnable.synchronizationValidation /* || other flags taht require VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME */)
        {
            insertToFeatureSetIfAvailable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME, "synchronizationValidation");
        }
        if (featuresToEnable.debugUtils)
        {
            insertToFeatureSetIfAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, "debugUtils");
        }
        if(featuresToEnable.swapchainMode.hasFlags(E_SWAPCHAIN_MODE::ESM_SURFACE))
        {
            insertToFeatureSetIfAvailable(VK_KHR_SURFACE_EXTENSION_NAME, "E_SWAPCHAIN_MODE::ESM_SURFACE flag for featureName");
#if defined(_NBL_PLATFORM_WINDOWS_)
            insertToFeatureSetIfAvailable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, "E_SWAPCHAIN_MODE::ESM_SURFACE flag for featureName");
#endif
        }
        SFeatures enabledFeatures = featuresToEnable;
        patchDependencies(selectedFeatureSet, enabledFeatures);

        const size_t totalFeatureCount = selectedFeatureSet.size();
        core::vector<const char*> extensionStringsToEnable(totalFeatureCount);
        uint32_t k = 0u;
        for (const auto& feature : selectedFeatureSet)
            extensionStringsToEnable[k++] = feature.c_str();

        if(!allRequestedFeaturesSupported)
            return nullptr;


        VkBaseInStructure* structsTail = nullptr;
        VkBaseInStructure* structsHead = nullptr;
        // Vulkan has problems with having features in the feature chain that have all values set to false.
        // For example having an empty "RayTracingPipelineFeaturesKHR" in the chain will lead to validation errors for RayQueryONLY applications.
        auto addStructToChain = [&structsHead, &structsTail](void* feature) -> void
        {
            VkBaseInStructure* toAdd = reinterpret_cast<VkBaseInStructure*>(feature);

            // For protecting against duplication of feature structures that may be requested to add to chain twice due to extension requirements
            const bool alreadyAdded = (toAdd->pNext != nullptr || toAdd == structsTail);

            if (structsHead == nullptr)
            {
                structsHead = toAdd;
                structsTail = toAdd;
            }
            else if (!alreadyAdded)
            {
                structsTail->pNext = toAdd;
                structsTail = toAdd;
            }
        };


        VkValidationFeaturesEXT validationFeaturesEXT = { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, nullptr };
        VkValidationFeatureEnableEXT validationsEnable[16u] = {};
        VkValidationFeatureDisableEXT validationsDisable[16u] = {};
        validationFeaturesEXT.pEnabledValidationFeatures = validationsEnable;

        // TODO: Do the samefor other validation features as well(?)
        if (enabledFeatures.synchronizationValidation)
        {
            validationsEnable[validationFeaturesEXT.enabledValidationFeatureCount] = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
            validationFeaturesEXT.enabledValidationFeatureCount += 1u;
            addStructToChain(&validationFeaturesEXT);
        }
        
        uint32_t instanceApiVersion = MinimumVulkanApiVersion;
        vkEnumerateInstanceVersion(&instanceApiVersion); // Get Highest
        if(instanceApiVersion < MinimumVulkanApiVersion)
        {
            assert(false);
            return nullptr;
        }

        std::unique_ptr<CVulkanDebugCallback> debugCallback = nullptr;
        VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, nullptr };
        if (logger && enabledFeatures.debugUtils)
        {
            auto logLevelMask = logger->getLogLevelMask();
            debugCallback = std::make_unique<CVulkanDebugCallback>(std::move(logger));

            debugMessengerCreateInfo.flags = 0;
            auto debugCallbackFlags = getDebugCallbackFlagsFromLogLevelMask(logLevelMask);
            debugMessengerCreateInfo.messageSeverity = debugCallbackFlags.first;
            debugMessengerCreateInfo.messageType = debugCallbackFlags.second;
            debugMessengerCreateInfo.pfnUserCallback = CVulkanDebugCallback::defaultCallback;
            debugMessengerCreateInfo.pUserData = debugCallback.get();

            addStructToChain(&debugMessengerCreateInfo);
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
            createInfo.pNext = structsHead;
            createInfo.flags = static_cast<VkInstanceCreateFlags>(0);
            createInfo.pApplicationInfo = &applicationInfo;
            createInfo.enabledLayerCount = requiredLayerNameCount;
            createInfo.ppEnabledLayerNames = requiredLayerNames;
            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionStringsToEnable.size());
            createInfo.ppEnabledExtensionNames = extensionStringsToEnable.data();

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

        CVulkanConnection* apiRaw = new CVulkanConnection(vk_instance, enabledFeatures, std::move(debugCallback), vk_debugMessenger);
        core::smart_refctd_ptr<CVulkanConnection> api(apiRaw, core::dont_grab);
        auto& physicalDevices = api->m_physicalDevices;
        physicalDevices.reserve(physicalDeviceCount);
        for (uint32_t i = 0u; i < physicalDeviceCount; ++i)
        {
            physicalDevices.emplace_back(std::make_unique<CVulkanPhysicalDevice>(
                core::smart_refctd_ptr(sys), api.get(), api->m_rdoc_api, vk_physicalDevices[i], vk_instance, instanceApiVersion));

        }

        return api;
    }

    CVulkanConnection::CVulkanConnection(
        VkInstance instance,
        const SFeatures& enabledFeatures,
        std::unique_ptr<CVulkanDebugCallback>&& debugCallback,
        VkDebugUtilsMessengerEXT vk_debugMessenger)
        : IAPIConnection(enabledFeatures)
        , m_vkInstance(instance)
        , m_debugCallback(std::move(debugCallback))
        , m_vkDebugUtilsMessengerEXT(vk_debugMessenger)
    {}

    CVulkanConnection::~CVulkanConnection()
    {
        if (m_vkDebugUtilsMessengerEXT != VK_NULL_HANDLE)
            vkDestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugUtilsMessengerEXT, nullptr);

        vkDestroyInstance(m_vkInstance, nullptr);
    }

    IDebugCallback* CVulkanConnection::getDebugCallback() const { return m_debugCallback.get(); }
}