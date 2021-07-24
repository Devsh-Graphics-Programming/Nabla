#include "nbl/video/surface/CSurfaceVKWin32.h"

#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

core::smart_refctd_ptr<CSurfaceVKWin32> CSurfaceVKWin32::create(const IAPIConnection* api, SCreationParams&& params)
{
    const CVulkanConnection* vk_connection = static_cast<const CVulkanConnection*>(api);

    return core::make_smart_refctd_ptr<CSurfaceVKWin32>(
        core::smart_refctd_ptr<const CVulkanConnection>(vk_connection), std::move(params));
    
}

CSurfaceVKWin32::CSurfaceVKWin32(core::smart_refctd_ptr<const CVulkanConnection>&& connection,
    SCreationParams&& params) : ISurfaceWin32(std::move(params)), ISurfaceVK(std::move(connection))
{
    VkWin32SurfaceCreateInfoKHR ci;
    ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    ci.hinstance = m_params.hinstance;
    ci.hwnd = m_params.hwnd;
    ci.flags = 0;
    ci.pNext = nullptr;
    vkCreateWin32SurfaceKHR(m_apiConnection->getInternalObject(), &ci, nullptr, &m_surface);
}

}