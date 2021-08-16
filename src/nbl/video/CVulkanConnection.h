#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommon.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
// #   include "nbl/video/surface/CSurfaceVKWin32.h"
#endif

namespace nbl::video
{

class CVulkanConnection final : public IAPIConnection
{
public:
    static core::smart_refctd_ptr<CVulkanConnection> create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        bool enableValidation = true)
    {
        const std::vector<const char*> requiredInstanceLayerNames({ "VK_LAYER_KHRONOS_validation" });

        // Todo(achal): Do I need to check availability for these?
        // Currently all applications are forced to have support for windows and debug utils messenger
        const uint32_t instanceExtensionCount = 3u;
        const char* instanceExtensions[instanceExtensionCount] = { "VK_KHR_surface", "VK_KHR_win32_surface", VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

        if (volkInitialize() != VK_SUCCESS)
        {
            printf("Failed to initialize volk!\n");
            return nullptr;
        }

        VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        applicationInfo.pNext = nullptr; // pNext must be NULL
        applicationInfo.pApplicationName = appName;
        applicationInfo.applicationVersion = appVer;
        applicationInfo.pEngineName = "Nabla";
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        applicationInfo.engineVersion = NABLA_VERSION_INTEGER;

        if (enableValidation)
        {
            if (!areAllInstanceLayersAvailable(requiredInstanceLayerNames))
            {
                printf("Validation layers requested but not available!\n");
                return nullptr;
            }
        }

        // Note(achal): This exact create info is used for application wide debug messenger for now
        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        debugUtilsMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugUtilsMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugUtilsMessengerCreateInfo.pfnUserCallback = &placeholderDebugCallback;

        VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugUtilsMessengerCreateInfo;
        instanceCreateInfo.flags = static_cast<VkInstanceCreateFlags>(0);
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(requiredInstanceLayerNames.size());
        instanceCreateInfo.ppEnabledLayerNames = requiredInstanceLayerNames.data();
        instanceCreateInfo.enabledExtensionCount = 3u;
        instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions;

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

        physical_devs_array_t physicalDevices = core::make_refctd_dynamic_array<physical_devs_array_t>(physicalDeviceCount);
        for (uint32_t i = 0u; i < physicalDeviceCount; ++i)
        {
            (*physicalDevices)[i] = core::make_smart_refctd_ptr<CVulkanPhysicalDevice>(
                vk_physicalDevices[i], core::smart_refctd_ptr(sys),
                core::make_smart_refctd_ptr<asset::IGLSLCompiler>(sys.get()));
        }

        return core::make_smart_refctd_ptr<CVulkanConnection>(vk_instance, physicalDevices, vk_debugMessenger);
    }

    VkInstance getInternalObject() const { return m_vkInstance; }

    core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const override
    {
        return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ m_physicalDevices->begin(), m_physicalDevices->end() };
    }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    // Todo(achal)
    IDebugCallback* getDebugCallback() const override { return nullptr; }

// Todo(achal): Remove
// private:

    using physical_devs_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IPhysicalDevice>>;

    CVulkanConnection(VkInstance instance, physical_devs_array_t physicalDevices,
        VkDebugUtilsMessengerEXT debugUtilsMessenger = VK_NULL_HANDLE)
        : m_vkInstance(instance), m_physicalDevices(physicalDevices),
        m_vkDebugUtilsMessengerEXT(debugUtilsMessenger)
    {}

    ~CVulkanConnection()
    {
        vkDestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugUtilsMessengerEXT, nullptr);
        vkDestroyInstance(m_vkInstance, nullptr);
    }

    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_vkDebugUtilsMessengerEXT = VK_NULL_HANDLE;
    physical_devs_array_t m_physicalDevices = nullptr;

    static VKAPI_ATTR VkBool32 VKAPI_CALL placeholderDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData)
    {
        printf("Validation Layer: %s\n", callbackData->pMessage);
        return VK_FALSE;
    }

    static inline bool areAllInstanceLayersAvailable(const std::vector<const char*>& requiredInstanceLayerNames)
    {
        uint32_t instanceLayerCount;
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
        std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

        for (const auto& requiredLayerName : requiredInstanceLayerNames)
        {
            const auto& result = std::find_if(instanceLayers.begin(), instanceLayers.end(),
                [requiredLayerName](const VkLayerProperties& layer) -> bool
                {
                    return (strcmp(requiredLayerName, layer.layerName) == 0);
                });

            if (result == instanceLayers.end())
                return false;
        }

        return true;
    }
};

}

#endif
