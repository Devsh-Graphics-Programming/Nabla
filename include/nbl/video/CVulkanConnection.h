#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
#endif

namespace nbl::video
{

class CVulkanConnection final : public IAPIConnection
{
public:
    static core::smart_refctd_ptr<CVulkanConnection> create(
        core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName,
        bool enableValidation = true);

    VkInstance getInternalObject() const { return m_vkInstance; }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    // Todo(achal)
    IDebugCallback* getDebugCallback() const override { return nullptr; }

// Todo(achal): Remove
// private:

    explicit CVulkanConnection(VkInstance instance, VkDebugUtilsMessengerEXT debugUtilsMessenger = VK_NULL_HANDLE)
        : IAPIConnection(), m_vkInstance(instance), m_vkDebugUtilsMessengerEXT(debugUtilsMessenger)
    {}

    virtual ~CVulkanConnection();

    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_vkDebugUtilsMessengerEXT = VK_NULL_HANDLE;

    static VKAPI_ATTR VkBool32 VKAPI_CALL placeholderDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData)
    {
        printf("Validation Layer: %s\n", callbackData->pMessage);
        return VK_FALSE;
    }

    static inline bool areAllInstanceLayersAvailable(const std::vector<const char*>& requiredInstanceLayerNames);
};

}

#endif
