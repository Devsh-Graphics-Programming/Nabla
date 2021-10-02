#include "nbl/video/CVulkanConnection.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommon.h"

namespace nbl::video
{
    core::smart_refctd_ptr<CVulkanConnection> CVulkanConnection::create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, bool enableValidation)
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

        auto api = core::make_smart_refctd_ptr<CVulkanConnection>(vk_instance, vk_debugMessenger);
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

    CVulkanConnection::~CVulkanConnection()
    {
        vkDestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugUtilsMessengerEXT, nullptr);
        vkDestroyInstance(m_vkInstance, nullptr);
    }

    bool CVulkanConnection::areAllInstanceLayersAvailable(const std::vector<const char*>& requiredInstanceLayerNames)
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
}