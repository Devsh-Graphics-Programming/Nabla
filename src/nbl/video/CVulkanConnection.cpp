#include "nbl/video/CVulkanConnection.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/debug/CVulkanDebugCallback.h"

// TODO: move inside `create` and call it LOG_FAIL and return nullptr
#define LOG(logger, ...) if (logger) {logger->log(__VA_ARGS__);}

namespace nbl::video
{

core::smart_refctd_ptr<CVulkanConnection> CVulkanConnection::create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, core::smart_refctd_ptr<system::ILogger>&& logger, const SFeatures& featuresToEnable)
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

    // TODO: its okay to use vectors here
    constexpr uint32_t MAX_LAYER_COUNT = 100u;
    constexpr uint32_t MAX_EXTENSION_COUNT = (1u << 12) / sizeof(char*);

    const size_t memSizeNeeded = MAX_EXTENSION_COUNT * sizeof(VkExtensionProperties) + MAX_LAYER_COUNT * sizeof(VkLayerProperties);
    void* mem = _NBL_ALIGNED_MALLOC(memSizeNeeded, _NBL_SIMD_ALIGNMENT);
    auto memFree = core::makeRAIIExiter([mem] {_NBL_ALIGNED_FREE(mem); });

    VkExtensionProperties* availableExtensions = static_cast<VkExtensionProperties*>(mem);
    VkLayerProperties* availableLayers = reinterpret_cast<VkLayerProperties*>(availableExtensions+MAX_EXTENSION_COUNT);

    // Get available layers
    uint32_t availableLayerCount = 0u;
    if (!getAvailableLayers(availableLayerCount,availableLayers))
        return nullptr;
    assert(availableLayerCount <= MAX_LAYER_COUNT);

    const char* requiredLayerNames[MAX_LAYER_COUNT] = { nullptr };
    uint32_t requiredLayerNameCount = 0u;
    {
        if (featuresToEnable.validations)
            requiredLayerNames[requiredLayerNameCount++] = "VK_LAYER_KHRONOS_validation";
        // RFC: why is validation required but not the others?
    }
    assert(requiredLayerNameCount <= MAX_LAYER_COUNT);

    const bool layersSupported = std::all_of(requiredLayerNames,requiredLayerNames+requiredLayerNameCount,
        [availableLayers,availableLayerCount,&logger](const char* layerName)
        {
            const VkLayerProperties* retval = std::find_if(availableLayers,availableLayers+availableLayerCount,
                [layerName](const VkLayerProperties& layerProps)
                {
                    return strcmp(layerName,layerProps.layerName)==0;
                }
            );

            if (retval == (availableLayers+availableLayerCount))
            {
                LOG(logger, "Failed to find required instance layer: %s\n", system::ILogger::ELL_ERROR, layerName);
                return false;
            }

            return true;
        }
    );
    if (!layersSupported)
        return nullptr;

    using FeatureSetType = core::unordered_set<core::string>;
    const FeatureSetType availableFeatureSet = [&]()->FeatureSetType
    {
        auto getExtensionsForLayer = [](const char* layerName, uint32_t& extensionCount, VkExtensionProperties* const extensions) -> bool
        {
            VkResult retval = vkEnumerateInstanceExtensionProperties(layerName,&extensionCount,nullptr);
            if (retval!=VK_SUCCESS && retval!=VK_INCOMPLETE)
                return false;

            if (extensions)
            {
                retval = vkEnumerateInstanceExtensionProperties(layerName,&extensionCount,extensions);
                if (retval!=VK_SUCCESS && retval!=VK_INCOMPLETE)
                    return false;
            }

            return true;
        };

        uint32_t totalCount = 0u;
        uint32_t count;

        if (getExtensionsForLayer(nullptr,count,availableExtensions))
            totalCount += count;
        else
            LOG(logger,"Failed to get implicit instance extensions!\n");

        for (uint32_t i=0u; i<requiredLayerNameCount; ++i)
        {
            if (getExtensionsForLayer(requiredLayerNames[i],count,availableExtensions+totalCount))
                totalCount += count;
            else
                LOG(logger, "Failed to get instance extensions for the layer: %s\n",system::ILogger::ELL_ERROR,requiredLayerNames[i]);
        }

        FeatureSetType retval;
        for (uint32_t i=0; i<totalCount; ++i)
            retval.insert(availableExtensions[i].extensionName);
        return retval;
    }();

    FeatureSetType selectedFeatureSet;
    bool allRequestedFeaturesSupported = true;
    auto insertToFeatureSetIfAvailable = [&](const char* extStr, const char* featureName) -> void
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
        insertToFeatureSetIfAvailable(VK_KHR_SURFACE_EXTENSION_NAME, "E_SWAPCHAIN_MODE::ESM_SURFACE flag for swapchainMode");
        insertToFeatureSetIfAvailable(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, "Obtaining colorspace from Swapchain");
        insertToFeatureSetIfAvailable(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, "Surface capabilities and formats query with pNext chains");
        // TODO: https://github.com/Devsh-Graphics-Programming/Nabla/issues/508
    #if defined(_NBL_PLATFORM_WINDOWS_)
        insertToFeatureSetIfAvailable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, "Win32 implementation/support for KHR_surface");
    #endif
    }
    SFeatures enabledFeatures = featuresToEnable;
    patchDependencies(selectedFeatureSet,enabledFeatures);

    const size_t totalFeatureCount = selectedFeatureSet.size();
    core::vector<const char*> extensionStringsToEnable;
    extensionStringsToEnable.reserve(totalFeatureCount);
    for (const auto& feature : selectedFeatureSet)
        extensionStringsToEnable.push_back(feature.c_str());

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

    // TODO: Do the same for other validation features as well(?)
    if (enabledFeatures.synchronizationValidation)
    {
        validationsEnable[validationFeaturesEXT.enabledValidationFeatureCount++] = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        addStructToChain(&validationFeaturesEXT);
    }
        
    uint32_t instanceApiVersion = MinimumVulkanApiVersion;
    vkEnumerateInstanceVersion(&instanceApiVersion); // Get Highest
    if(instanceApiVersion<MinimumVulkanApiVersion)
    {
        LOG(logger,"Vulkan Instance version too low %d vs the required %d",system::ILogger::ELL_ERROR,instanceApiVersion,MinimumVulkanApiVersion);
        return nullptr;
    }
    // TODO: should probably clamp the `instanceApiVersion` to not require Vulkan 2.0 or something XD

    // speculatively create a debug callback so we don't need to pick between loggers later on
    const auto logLevelMask = logger->getLogLevelMask();
    std::unique_ptr<CVulkanDebugCallback> debugCallback = std::make_unique<CVulkanDebugCallback>(std::move(logger));

    VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, nullptr };
    if (enabledFeatures.debugUtils)
    {
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

        if (vkCreateInstance(&createInfo,nullptr,&vk_instance)!=VK_SUCCESS)
            return nullptr;
    }

    volkLoadInstanceOnly(vk_instance);

    core::vector<VkPhysicalDevice> vk_physicalDevices; vk_physicalDevices.reserve(16);
    {
        uint32_t physicalDeviceCount = 0u;
        VkResult retval = vkEnumeratePhysicalDevices(vk_instance,&physicalDeviceCount,nullptr);
        if (retval!=VK_SUCCESS && retval!=VK_INCOMPLETE)
        {
            if (debugCallback)
                LOG(debugCallback->getLogger(), "Failed to enumerate physical devices!\n");
            return nullptr;
        }
        vk_physicalDevices.resize(physicalDeviceCount);
        vkEnumeratePhysicalDevices(vk_instance,&physicalDeviceCount,vk_physicalDevices.data());
    }

    VkDebugUtilsMessengerEXT vk_debugMessenger = VK_NULL_HANDLE;
    if (debugCallback)
    {
        if (vkCreateDebugUtilsMessengerEXT(vk_instance,&debugMessengerCreateInfo,nullptr,&vk_debugMessenger) != VK_SUCCESS)
        {
            LOG(debugCallback->getLogger(),"Failed to create debug utils messenger!\n");
            return nullptr;
        }
    }

    core::smart_refctd_ptr<CVulkanConnection> api(new CVulkanConnection(vk_instance,enabledFeatures,std::move(debugCallback),vk_debugMessenger),core::dont_grab);
    api->m_physicalDevices.reserve(vk_physicalDevices.size());
    for (auto vk_physicalDevice : vk_physicalDevices)
    {
        auto device = CVulkanPhysicalDevice::create(core::smart_refctd_ptr(sys),api.get(),api->m_rdoc_api,vk_physicalDevice);
        if (!device)
        {
            LOG(api->getDebugCallback()->getLogger(), "Vulkan device %p found but doesn't meet minimum Nabla requirements. Skipping!", system::ILogger::ELL_WARNING, vk_physicalDevice);
            continue;
        }
        api->m_physicalDevices.emplace_back(std::move(device));
    }
#undef LOF

    // TODO: should we return created non-null API connections that have 0 physical devices?
    return api;
}

CVulkanConnection::~CVulkanConnection()
{
    if (m_vkDebugUtilsMessengerEXT!=VK_NULL_HANDLE)
        vkDestroyDebugUtilsMessengerEXT(m_vkInstance,m_vkDebugUtilsMessengerEXT,nullptr);

    vkDestroyInstance(m_vkInstance,nullptr);
}

}
