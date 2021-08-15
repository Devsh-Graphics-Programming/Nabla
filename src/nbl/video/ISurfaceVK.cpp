#include "nbl/video/surface/ISurfaceVK.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanConnection.h"

#include <volk.h>

namespace nbl::video
{

template <typename Window>
core::smart_refctd_ptr<CSurfaceVulkan<Window>> CSurfaceVulkan<Window>::create(
    core::smart_refctd_ptr<video::CVulkanConnection>&& api,
    core::smart_refctd_ptr<Window>&& window)
{
    if (!api || !window)
        return nullptr;

    if constexpr (std::is_same_v<Window, ui::IWindowWin32>)
    {
        VkWin32SurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
        createInfo.pNext = nullptr; // pNext must be NULL
        createInfo.flags = static_cast<VkWin32SurfaceCreateFlagsKHR>(0);
        createInfo.hinstance = GetModuleHandle(NULL);
        createInfo.hwnd = (static_cast<HWND>(window->getNativeHandle()));

        VkSurfaceKHR vk_surface;
        if (vkCreateWin32SurfaceKHR(api->getInternalObject(), &createInfo, nullptr,
            &vk_surface) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<this_t>(std::move(api), std::move(window), vk_surface);
        }
        else
        {
            return nullptr;
        }
    }
}

template <typename Window>
CSurfaceVulkan<Window>::~CSurfaceVulkan()
{
    if (base_t::m_api->getAPIType() == EAT_VULKAN)
    {
        VkInstance vk_instance = static_cast<const CVulkanConnection*>(base_t::m_api.get())->getInternalObject();
        vkDestroySurfaceKHR(vk_instance, m_surface, nullptr);
    }
}

template class CSurfaceVulkan<ui::IWindowWin32>;

#if 0
ISurfaceVK::ISurfaceVK(core::smart_refctd_ptr<const CVulkanConnection>&& apiConnection)
    : m_apiConnection(std::move(apiConnection)) {}

bool ISurfaceVK::isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const
{
    if (dev->getAPIType() != EAT_VULKAN)
    {
        // Todo(achal): Log error
        return false;
    }

    auto vkphd = static_cast<const CVulkanPhysicalDevice*>(dev)->getInternalObject();
    VkBool32 supported;
    vkGetPhysicalDeviceSurfaceSupportKHR(vkphd, _queueFamIx, m_surface, &supported);

    return static_cast<bool>(supported);
}

ISurfaceVK::~ISurfaceVK()
{
    vkDestroySurfaceKHR(m_apiConnection->getInternalObject(), m_surface, nullptr);
}
#endif

}