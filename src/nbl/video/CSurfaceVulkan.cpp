#include "nbl/video/surface/CSurfaceVulkan.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanConnection.h"

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
        vkDestroySurfaceKHR(vk_instance, m_vkSurfaceKHR, nullptr);
    }
}

template <typename Window>
bool CSurfaceVulkan<Window>::isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const
{
    if (dev->getAPIType() != EAT_VULKAN)
        return false;

    VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(dev)->getInternalObject();

    VkBool32 supported;
    if (vkGetPhysicalDeviceSurfaceSupportKHR(vk_physicalDevice, _queueFamIx, m_vkSurfaceKHR, &supported) == VK_SUCCESS)
    {
        return static_cast<bool>(supported);
    }
    else
    {
        return false;
    }
}

template class CSurfaceVulkan<ui::IWindowWin32>;

}