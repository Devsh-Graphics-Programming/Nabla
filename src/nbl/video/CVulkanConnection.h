#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IAPIConnection.h"
// #include "nbl/video/surface/CSurfaceVKWin32.h"
// #include "nbl/video/CVulkanPhysicalDevice.h"

namespace nbl {
namespace video
{
class CVulkanConnection final : public IAPIConnection
{
public:
    CVulkanConnection(uint32_t appVer, const char* appName, const SDebugCallback& dbgCb) : IAPIConnection(dbgCb)
    {
        VkResult result = volkInitialize();
        assert(result == VK_SUCCESS);
        
        VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        applicationInfo.engineVersion = NABLA_VERSION_INTEGER;
        applicationInfo.pEngineName = "Nabla";
        applicationInfo.applicationVersion = appVer;
        applicationInfo.pApplicationName = appName;

        VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        
        vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance);
        assert(m_instance);
        
        // volkLoadInstanceOnly(m_instance);
        volkLoadInstance(m_instance);
        
        // uint32_t devCount = 0u;
        // vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr);
        // core::vector<VkPhysicalDevice> vkphds(devCount, VK_NULL_HANDLE);
        // vkEnumeratePhysicalDevices(m_instance, &devCount, vkphds.data());

        // m_physDevices = core::make_refctd_dynamic_array<physical_devs_array_t>(devCount);
        // for (uint32_t i = 0u; i < devCount; ++i)
        // {
        //     (*m_physDevices)[i] = core::make_smart_refctd_ptr<CVulkanPhysicalDevice>(vkphds[i]);
        // }
    }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const override
    {
        return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ m_physDevices->begin(), m_physDevices->end() };
    }

    core::smart_refctd_ptr<ISurface> createSurface(ui::IWindow* window) const override
    {
        return nullptr;
    }

    VkInstance getInternalObject() const { return m_instance; }

protected:
    ~CVulkanConnection()
    {
        vkDestroyInstance(m_instance, nullptr);
    }

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    using physical_devs_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IPhysicalDevice>>;
    physical_devs_array_t m_physDevices;
};

}
}


#endif
