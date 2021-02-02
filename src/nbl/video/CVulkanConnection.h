#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IAPIConnection.h"
#include "nbl/video/surface/CSurfaceVKWin32.h"
#include "nbl/video/CVulkanPhysicalDevice.h"

namespace nbl {
namespace video
{

class CVulkanConnection final : public IAPIConnection
{
public:
    CVulkanConnection(uint32_t appVer, const char* appName)
    {
        VkResult result = volkInitialize();
        assert(result == VK_SUCCESS);

        VkApplicationInfo app;
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        app.engineVersion = NABLA_VERSION_INTEGER;
        app.pEngineName = "Nabla";
        app.applicationVersion = appVer;
        app.pApplicationName = appName;
        app.pNext = nullptr;
        VkInstanceCreateInfo ci;
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.flags = 0;
        ci.enabledExtensionCount = 0;
        ci.ppEnabledExtensionNames = nullptr;
        ci.enabledLayerCount = 0;
        ci.ppEnabledLayerNames = nullptr;
        ci.pNext = nullptr;
        ci.pApplicationInfo = &app;

        vkCreateInstance(&ci, nullptr, &m_instance);

        volkLoadInstanceOnly(m_instance);

        uint32_t devCount = 0u;
        vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr);
        core::vector<VkPhysicalDevice> vkphds(devCount, VK_NULL_HANDLE);
        vkEnumeratePhysicalDevices(m_instance, &devCount, vkphds.data());

        m_physDevices = core::make_refctd_dynamic_array<physical_devs_array_t>(devCount);
        for (uint32_t i = 0u; i < devCount; ++i)
        {
            (*m_physDevices)[i] = core::make_smart_refctd_ptr<CVulkanPhysicalDevice>(vkphds[i]);
        }
    }

    E_TYPE getAPIType() const override { return ET_VULKAN; }

    core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const override
    {
        return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ m_physDevices->begin(), m_physDevices->end() };
    }

    VkInstance getInternalObject() const { return m_instance; }

protected:
    ~CVulkanConnection()
    {
        vkDestroyInstance(m_instance, nullptr);
    }

private:
    VkInstance m_instance = nullptr;
    using physical_devs_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IPhysicalDevice>>;
    physical_devs_array_t m_physDevices;
};

}
}

#endif
