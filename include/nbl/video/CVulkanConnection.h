#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/debug/CVulkanDebugCallback.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
#endif

#include <volk/volk.h>

namespace nbl::video
{
class NBL_API2 CVulkanConnection final : public IAPIConnection
{
public:
    static core::smart_refctd_ptr<CVulkanConnection> create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        core::smart_refctd_ptr<system::ILogger>&& logger, const SFeatures& featuresToEnable);

    VkInstance getInternalObject() const { return m_vkInstance; }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    IDebugCallback* getDebugCallback() const override;

protected:
    explicit CVulkanConnection(
        VkInstance instance,
        const SFeatures& enabledFeatures,
        std::unique_ptr<CVulkanDebugCallback>&& debugCallback,
        VkDebugUtilsMessengerEXT vk_debugMessenger);

    virtual ~CVulkanConnection();

private:
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

    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_vkDebugUtilsMessengerEXT = VK_NULL_HANDLE;
    std::unique_ptr<CVulkanDebugCallback> m_debugCallback = nullptr; // this needs to live longer than VkDebugUtilsMessengerEXT handle above
};

}

#endif
