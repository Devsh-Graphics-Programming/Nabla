#ifndef _NBL_C_VULKAN_CONNECTION_H_INCLUDED_
#define _NBL_C_VULKAN_CONNECTION_H_INCLUDED_

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
            core::smart_refctd_ptr<system::ILogger>&& logger, const SFeatures& featuresToEnable
        );

        inline VkInstance getInternalObject() const {return m_vkInstance;}

        inline E_API_TYPE getAPIType() const override {return EAT_VULKAN;}

        inline IDebugCallback* getDebugCallback() const override {return m_debugCallback.get();}

    protected:
        explicit inline CVulkanConnection(const VkInstance instance, const SFeatures& enabledFeatures, std::unique_ptr<CVulkanDebugCallback>&& debugCallback, const VkDebugUtilsMessengerEXT vk_debugMessenger)
            : IAPIConnection(enabledFeatures), m_vkInstance(instance), m_debugCallback(std::move(debugCallback)), m_vkDebugUtilsMessengerEXT(vk_debugMessenger) {}

        virtual ~CVulkanConnection();

    private:
        const VkInstance m_vkInstance;
        const VkDebugUtilsMessengerEXT m_vkDebugUtilsMessengerEXT;
        const std::unique_ptr<CVulkanDebugCallback> m_debugCallback; // this needs to live longer than VkDebugUtilsMessengerEXT handle above
};

}

#endif
