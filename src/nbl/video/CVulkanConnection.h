#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video//debug/CVulkanDebugCallback.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
#endif

#define LOG(logger, ...) if (logger) {logger->log(__VA_ARGS__);}

namespace nbl::video
{

class CVulkanConnection final : public IAPIConnection
{
public:
    static core::smart_refctd_ptr<CVulkanConnection> create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        const uint32_t extensionCount, video::IAPIConnection::E_EXTENSION* extensions,
        core::smart_refctd_ptr<system::ILogger>&& logger, bool enableValidation)
    {
        if (volkInitialize() != VK_SUCCESS)
        {
            LOG(logger, "Failed to initialize volk!\n", system::ILogger::ELL_ERROR);
            return nullptr;
        }

        constexpr uint32_t MAX_EXTENSION_COUNT = (1u << 12) / sizeof(char*);
        constexpr uint32_t MAX_LAYER_COUNT = 100u;

        const size_t memSizeNeeded = MAX_EXTENSION_COUNT * sizeof(VkExtensionProperties) + MAX_LAYER_COUNT * sizeof(VkLayerProperties);
        void* mem = _NBL_ALIGNED_MALLOC(memSizeNeeded, _NBL_SIMD_ALIGNMENT);
        auto memFree = core::makeRAIIExiter([mem] {_NBL_ALIGNED_FREE(mem); });

        VkExtensionProperties* availableExtensions = static_cast<VkExtensionProperties*>(mem);
        VkLayerProperties* availableLayers = reinterpret_cast<VkLayerProperties*>(availableExtensions + MAX_EXTENSION_COUNT);

        const char* requiredExtensionNames[MAX_EXTENSION_COUNT];
        uint32_t requiredExtensionNameCount = 0u;
        {
            if (logger)
                requiredExtensionNames[requiredExtensionNameCount++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

            for (uint32_t i = 0u; i < extensionCount; ++i)
            {
                // Handle other platforms
                if (extensions[i] == video::IAPIConnection::E_SURFACE)
                {
                    requiredExtensionNames[requiredExtensionNameCount++] = "VK_KHR_surface";
                    requiredExtensionNames[requiredExtensionNameCount++] = "VK_KHR_win32_surface";
                }
            }
        }
        assert(requiredExtensionNameCount <= MAX_EXTENSION_COUNT);

        const char* requiredLayerNames[MAX_LAYER_COUNT] = { nullptr };
        uint32_t requiredLayerNameCount = 0u;
        {
            if (enableValidation)
            {
                requiredLayerNames[requiredLayerNameCount++] = "VK_LAYER_KHRONOS_validation";
            }
        }
        assert(requiredLayerNameCount <= MAX_LAYER_COUNT);

        uint32_t availableExtensionCount = 0u;
        {
            uint32_t count;

            if (!getExtensionsForLayer(nullptr, count, availableExtensions))
            {
                LOG(logger, "Failed to get implicit instance extensions!\n");
                return nullptr;
            }

            availableExtensionCount += count;

            for (uint32_t i = 0u; i < requiredLayerNameCount; ++i)
            {
                if (!getExtensionsForLayer(requiredLayerNames[i], count, availableExtensions + availableExtensionCount))
                {
                    LOG(logger, "Failed to get instance extensions for the layer: %s\n", system::ILogger::ELL_ERROR, requiredLayerNames[i]);
                    return nullptr;
                }

                availableExtensionCount += count;
            }
        }
        assert(availableExtensionCount <= MAX_EXTENSION_COUNT);

        uint32_t availableLayerCount = 0u;
        {
            uint32_t count;
            VkResult retval = vkEnumerateInstanceLayerProperties(&count, nullptr);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
                return nullptr;

            retval = vkEnumerateInstanceLayerProperties(&count, availableLayers);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
                return nullptr;

            availableLayerCount += count;
        }
        assert(availableLayerCount <= MAX_LAYER_COUNT);

        // Can't find anything in C++ STL for this
        const bool extensionsSupported = std::all_of(requiredExtensionNames, requiredExtensionNames + requiredExtensionNameCount,
            [availableExtensions, availableExtensionCount, &logger](const char* extensionName)
            {
                const VkExtensionProperties* retval = std::find_if(availableExtensions, availableExtensions + availableExtensionCount,
                    [extensionName](const VkExtensionProperties& extensionProps)
                    {
                        return strcmp(extensionName, extensionProps.extensionName) == 0;
                    });

                if (retval == (availableExtensions + availableExtensionCount))
                {
                    LOG(logger, "Failed to find required instance extension: %s\n", system::ILogger::ELL_ERROR, extensionName);
                    return false;
                }

                return true;
            });

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

        if (!extensionsSupported || !layersSupported)
            return nullptr;

#if 0
        // Create a CVulkanDebugCallback here based on if logger is available
        core::smart_refctd_ptr<CVulkanDebugCallback> debugCallback = nullptr;
        if (logger)
        {
            // If the logger is available I would need the instance extension VK_EXT_DEBUG_UTILS_EXTENSION_NAME
            // to create a debug callback so first check the availability of this instance extension
            VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
            const void* pNext;
            VkDebugUtilsMessengerCreateFlagsEXT     flags;
            VkDebugUtilsMessageSeverityFlagsEXT     messageSeverity;
            VkDebugUtilsMessageTypeFlagsEXT         messageType;
            PFN_vkDebugUtilsMessengerCallbackEXT    pfnUserCallback;
            void* pUserData;
        }

        if (debugCallback && enableValidation)
        {
            if (!isInstanceLayerAvailable(validationLayer))
            {
                logger->log("Validation layer (%s) requested but not available!\n", system::ILogger::ELL_ERROR, validationLayer);
                return nullptr;
            }
        }
#endif


        VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        applicationInfo.pNext = nullptr; // pNext must be NULL
        applicationInfo.pApplicationName = appName;
        applicationInfo.applicationVersion = appVer;
        applicationInfo.pEngineName = "Nabla";
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        applicationInfo.engineVersion = NABLA_VERSION_INTEGER;

        // Just copy the logger->getLogLevelMasks here as well
        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
#if 0
        const void* pNext;
        VkDebugUtilsMessengerCreateFlagsEXT     flags;
        VkDebugUtilsMessageSeverityFlagsEXT     messageSeverity;
        VkDebugUtilsMessageTypeFlagsEXT         messageType;
        PFN_vkDebugUtilsMessengerCallbackEXT    pfnUserCallback;
        void* pUserData;
#endif

        debugUtilsMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugUtilsMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        // debugUtilsMessengerCreateInfo.pfnUserCallback = debugCallback.defaultCallback;
        debugUtilsMessengerCreateInfo.pfnUserCallback = placeholderDebugCallback;

        VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugUtilsMessengerCreateInfo;
        instanceCreateInfo.flags = static_cast<VkInstanceCreateFlags>(0);
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        instanceCreateInfo.enabledLayerCount = requiredLayerNameCount;
        instanceCreateInfo.ppEnabledLayerNames = requiredLayerNames;
        instanceCreateInfo.enabledExtensionCount = requiredExtensionNameCount;
        instanceCreateInfo.ppEnabledExtensionNames = requiredExtensionNames;

        VkInstance vk_instance;
        if (vkCreateInstance(&instanceCreateInfo, nullptr, &vk_instance) != VK_SUCCESS)
        {
            printf("Failed to create vulkan instance, for some reason!\n");
            return nullptr;
        }

        // Todo(achal): Perhaps use volkLoadInstanceOnly?
        volkLoadInstance(vk_instance);

        VkDebugUtilsMessengerEXT vk_debugMessenger;
        if (vkCreateDebugUtilsMessengerEXT(vk_instance, &debugUtilsMessengerCreateInfo, nullptr, &vk_debugMessenger) != VK_SUCCESS)
        {
            printf("Failed to create debug messenger for some reason!\n");
            return nullptr;
        }

        constexpr uint32_t MAX_PHYSICAL_DEVICE_COUNT = 16u;
        uint32_t physicalDeviceCount = 0u;
        VkPhysicalDevice vk_physicalDevices[MAX_PHYSICAL_DEVICE_COUNT];
        {
            VkResult retval = vkEnumeratePhysicalDevices(vk_instance, &physicalDeviceCount, nullptr);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            {
                printf("Failed to enumerate physical devices!\n");
                return nullptr;
            }

            vkEnumeratePhysicalDevices(vk_instance, &physicalDeviceCount, vk_physicalDevices);
        }

        auto api = core::make_smart_refctd_ptr<CVulkanConnection>(vk_instance,vk_debugMessenger);
        auto& physicalDevices = api->m_physicalDevices;
        physicalDevices.reserve(physicalDeviceCount);
        for (uint32_t i = 0u; i < physicalDeviceCount; ++i)
        {
            physicalDevices.emplace_back(std::make_unique<CVulkanPhysicalDevice>(
                core::smart_refctd_ptr(sys),
                core::make_smart_refctd_ptr<asset::IGLSLCompiler>(sys.get()),
                api.get(), vk_physicalDevices[i]
            ));
        }
        return api;
    }

    VkInstance getInternalObject() const { return m_vkInstance; }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    // Todo(achal)
    IDebugCallback* getDebugCallback() const override { return nullptr; }

// Todo(achal): Remove
// private:

    CVulkanConnection(VkInstance instance, VkDebugUtilsMessengerEXT debugUtilsMessenger = VK_NULL_HANDLE)
        : IAPIConnection(), m_vkInstance(instance), m_vkDebugUtilsMessengerEXT(debugUtilsMessenger)
    {}

    ~CVulkanConnection()
    {
        vkDestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugUtilsMessengerEXT, nullptr);
        vkDestroyInstance(m_vkInstance, nullptr);
    }

    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_vkDebugUtilsMessengerEXT = VK_NULL_HANDLE;

    static VKAPI_ATTR VkBool32 VKAPI_CALL placeholderDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData)
    {
        printf("Validation Layer: %s\n", callbackData->pMessage);
        return VK_FALSE;
    }

    static inline bool isInstanceLayerAvailable(const char* requiredInstanceLayer)
    {
        uint32_t instanceLayerCount;
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
        std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

        auto retval = std::find_if(instanceLayers.begin(), instanceLayers.end(),
            [requiredInstanceLayer](const VkLayerProperties& layer) -> bool
            {
                return (strcmp(requiredInstanceLayer, layer.layerName) == 0);
            });

        return retval != instanceLayers.end();
    }

    static inline bool getExtensionsForLayer(const char* layerName, uint32_t& extensionCount,
        VkExtensionProperties* extensions)
    {
        VkResult retval = vkEnumerateInstanceExtensionProperties(layerName, &extensionCount, nullptr);
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return false;

        if (extensions)
        {
            retval = vkEnumerateInstanceExtensionProperties(layerName, &extensionCount, extensions);
            if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
                return false;
        }

        return true;
    }
};

}

#endif
