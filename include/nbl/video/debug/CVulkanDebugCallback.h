#ifndef __NBL_VIDEO_C_VULKAN_DEBUG_CALLBACK_H_INCLUDED__
#define __NBL_VIDEO_C_VULKAN_DEBUG_CALLBACK_H_INCLUDED__

#include "nbl/video/debug/IDebugCallback.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace nbl::video
{

class CVulkanDebugCallback : public IDebugCallback
{
public:
    explicit CVulkanDebugCallback(core::smart_refctd_ptr<system::ILogger>&& _logger)
        : IDebugCallback(std::move(_logger)), m_callback(&defaultCallback)
    {}

    static VKAPI_ATTR VkBool32 VKAPI_CALL defaultCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData)
    {
        const auto* cb = reinterpret_cast<const CVulkanDebugCallback*>(userData);

        uint8_t level = 0;

        switch (messageSeverity)
        {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            level |= system::ILogger::ELL_INFO;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            level |= system::ILogger::ELL_WARNING;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            level |= system::ILogger::ELL_ERROR;
            break;
        default:
            assert(!"Don't know what to do with this value!");
        }

        switch (messageType)
        {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
            level |= system::ILogger::ELL_INFO;
            break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
            level |= system::ILogger::ELL_ERROR;
            break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
            level |= system::ILogger::ELL_PERFORMANCE;
            break;
        default:
            assert(!"Don't know what to do with this value!");
        }

        cb->getLogger()->log("%s", static_cast<system::ILogger::E_LOG_LEVEL>(level), callbackData->pMessage);

        return VK_FALSE;
    }

    decltype(&defaultCallback) m_callback;
};

}

#endif
