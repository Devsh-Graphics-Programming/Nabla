#include "nbl/video/surface/CSurfaceVKWin32.h"

#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{
#if 0
core::smart_refctd_ptr<CSurfaceVKWin32> CSurfaceVKWin32::create(const IAPIConnection* api,
    SCreationParams&& params)
{
    if (api->getAPIType() != EAT_VULKAN)
        return nullptr;
    const CVulkanConnection* vulkanConnection = static_cast<const CVulkanConnection*>(api);

    VkWin32SurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    createInfo.pNext = nullptr;
    createInfo.flags = static_cast<VkWin32SurfaceCreateFlagsKHR>(0);
    createInfo.hinstance = m_params.hinstance;
    createInfo.hwnd = m_params.hwnd;

    VkSurfaceKHR vk_surface;
    if (vkCreateWin32SurfaceKHR(vulkanConnection->getInternalObject(), &createInfo, nullptr, &vk_surface) == VK_SUCCESS)
    {
        return core::make_smart_refctd_ptr<CSurfaceVKWin32>();
    }
    else
    {
        return nullptr;
    }
}

CSurfaceVKWin32::CSurfaceVKWin32(core::smart_refctd_ptr<const CVulkanConnection>&& connection,
    SCreationParams&& params) : ISurfaceWin32(std::move(params)), ISurfaceVK(std::move(connection))
{
    
    vkCreateWin32SurfaceKHR(m_apiConnection->getInternalObject(), &ci, nullptr, &m_surface);
}
#endif

}