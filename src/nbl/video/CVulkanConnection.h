#ifndef __NBL_C_VULKAN_CONNECTION_H_INCLUDED__
#define __NBL_C_VULKAN_CONNECTION_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IAPIConnection.h"
#include "nbl/video/surface/CSurfaceVKWin32.h"

namespace nbl {
namespace video
{

class CVulkanConnection final : public IAPIConnection
{
public:
    CVulkanConnection()
    {
        VkResult result = volkInitialize();
        assert(result == VK_SUCCESS);

        VkApplicationInfo app;
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_MAKE_VERSION(1, 1, 0);
        app.engineVersion = NABLA_VERSION_INTEGER;
        app.pEngineName = "Nabla";
        app.applicationVersion = 0;
        app.pApplicationName = nullptr;
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
    }

    ~CVulkanConnection()
    {
        vkDestroyInstance(m_instance, nullptr);
    }

    E_TYPE getAPIType() const override { return ET_VULKAN; }

    VkInstance getInternalObject() const { return m_instance; }

private:
    VkInstance m_instance = nullptr;
};

}
}

#endif
