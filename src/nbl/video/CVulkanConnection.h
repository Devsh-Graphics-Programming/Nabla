#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CVulkanPhysicalDevice.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
#   include "nbl/video/surface/CSurfaceVKWin32.h"
#endif

namespace nbl::video
{

class CVulkanConnection final : public IAPIConnection
{
public:
    CVulkanConnection(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer,
        const char* appName, const SDebugCallback& dbgCb) : IAPIConnection(std::move(sys))
    {
        VkResult result = volkInitialize();
        assert(result == VK_SUCCESS);
        
        VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        applicationInfo.engineVersion = NABLA_VERSION_INTEGER;
        applicationInfo.pEngineName = "Nabla";
        applicationInfo.applicationVersion = appVer;
        applicationInfo.pApplicationName = appName;
        
        // Todo(achal): Get this from the user
        const std::vector<const char*> requiredInstanceLayerNames({ "VK_LAYER_KHRONOS_validation" });
        assert(areAllInstanceLayersAvailable(requiredInstanceLayerNames));

        // Note(achal): This exact create info is used for application wide debug messenger for now
        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        debugUtilsMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugUtilsMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugUtilsMessengerCreateInfo.pfnUserCallback = &placeholderDebugCallback;

        VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugUtilsMessengerCreateInfo;
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(requiredInstanceLayerNames.size());
        instanceCreateInfo.ppEnabledLayerNames = requiredInstanceLayerNames.data();

        {
            // Todo(achal): Get this from the user
            const bool isWorkloadHeadlessCompute = false;

            // Todo(achal): Not always use the debug messenger. There are also other extensions for this, check if we want to use them.
            if (isWorkloadHeadlessCompute)
            {
                const uint32_t instanceExtensionCount = 1u;
                const char* instanceExtensions[instanceExtensionCount] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

                instanceCreateInfo.enabledExtensionCount = instanceExtensionCount;
                instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions;
            }
            else
            {
                // Todo(achal): This needs to be handled in a platform agnostic way.
                const uint32_t instanceExtensionCount = 3u;
                char* instanceExtensions[instanceExtensionCount] = { "VK_KHR_surface", "VK_KHR_win32_surface", VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

                instanceCreateInfo.enabledExtensionCount = instanceExtensionCount;
                instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions;
            }
        }
        
        if (vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance) != VK_SUCCESS)
            printf("FAiled to create vulkan instance!\n");
        assert(m_instance);
        
        // Todo(achal): I gotta look into this ie if (and how) we want to use volkLoadInstanceOnly
        // volkLoadInstanceOnly(m_instance);
        volkLoadInstance(m_instance);

        vkCreateDebugUtilsMessengerEXT(m_instance, &debugUtilsMessengerCreateInfo, nullptr, &m_debugMessenger);
        assert(m_debugMessenger);
        
        uint32_t devCount = 0u;
        vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr);
        core::vector<VkPhysicalDevice> vkphds(devCount, VK_NULL_HANDLE);
        vkEnumeratePhysicalDevices(m_instance, &devCount, vkphds.data());

        m_physDevices = core::make_refctd_dynamic_array<physical_devs_array_t>(devCount);
        for (uint32_t i = 0u; i < devCount; ++i)
        {
            (*m_physDevices)[i] = core::make_smart_refctd_ptr<CVulkanPhysicalDevice>(vkphds[i], core::smart_refctd_ptr(m_system), std::move(m_GLSLCompiler));
        }
    }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const override
    {
        return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ m_physDevices->begin(), m_physDevices->end() };
    }

    core::smart_refctd_ptr<ISurface> createSurface(ui::IWindow* window) const override;

    VkInstance getInternalObject() const { return m_instance; }

protected:
    ~CVulkanConnection()
    {
        vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        vkDestroyInstance(m_instance, nullptr);
    }

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    using physical_devs_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IPhysicalDevice>>;
    physical_devs_array_t m_physDevices;

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
