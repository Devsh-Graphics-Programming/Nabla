#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
#endif

#include <volk/volk.h>

namespace nbl::video
{
class CVulkanDebugCallback;

class CVulkanConnection final : public IAPIConnection
{
public:
    static core::smart_refctd_ptr<CVulkanConnection> create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        const uint32_t requiredFeatureCount, video::IAPIConnection::E_FEATURE* requiredFeatures,
        const uint32_t optionalFeatureCount, video::IAPIConnection::E_FEATURE* optionalFeatures,
        core::smart_refctd_ptr<system::ILogger>&& logger, bool enableValidation = true);

    VkInstance getInternalObject() const { return m_vkInstance; }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    IDebugCallback* getDebugCallback() const override;

// Todo(achal): Remove
// private:

    explicit CVulkanConnection(VkInstance instance, std::unique_ptr<CVulkanDebugCallback>&& debugCallback,
        VkDebugUtilsMessengerEXT vk_debugMessenger);

    virtual ~CVulkanConnection();

    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_vkDebugUtilsMessengerEXT = VK_NULL_HANDLE;
    std::unique_ptr<CVulkanDebugCallback> m_debugCallback = nullptr; // this needs to live longer than VkDebugUtilsMessengerEXT handle above

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

    static inline void getVulkanExtensionNamesFromFeature(const IAPIConnection::E_FEATURE feature, uint32_t& extNameCount, const char** extNames)
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
};

}

#endif
