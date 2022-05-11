#ifndef __NBL_C_SURFACE_VULKAN_H_INCLUDED__
#define __NBL_C_SURFACE_VULKAN_H_INCLUDED__

#include "BuildConfigOptions.h"

#include "nbl/video/surface/ISurface.h"
#include "nbl/video/CVulkanConnection.h"

#include "nbl/ui/IWindowAndroid.h"

namespace nbl::video
{


#include "volk.h"

class NBL_API ISurfaceVulkan : public ISurface
{
    using base_t = ISurface;
public:
    bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override;
    void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const override;
    E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const override;
    bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const override;

    inline VkSurfaceKHR getInternalObject() const { return m_vkSurfaceKHR; }

protected:
    explicit ISurfaceVulkan(core::smart_refctd_ptr<video::IAPIConnection>&& api, VkSurfaceKHR vk_surface) : base_t(std::move(api)), m_vkSurfaceKHR(vk_surface) {}
    virtual ~ISurfaceVulkan() = default;

    VkSurfaceKHR m_vkSurfaceKHR = VK_NULL_HANDLE;
};

#ifdef _NBL_PLATFORM_WINDOWS_
class NBL_API CSurfaceVulkanWin32 final : public CSurface<ui::IWindowWin32, ISurfaceVulkan>
{
    using this_t = CSurfaceVulkanWin32;
    using base_t = CSurface<ui::IWindowWin32, ISurfaceVulkan>;
public:
    CSurfaceVulkanWin32(core::smart_refctd_ptr<ui::IWindowWin32>&& window, core::smart_refctd_ptr<IAPIConnection>&& api, VkSurfaceKHR surf) :
        base_t(std::move(window), std::move(api), surf)
    {
    }
    static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, core::smart_refctd_ptr<ui::IWindowWin32>&& window)
    {
        if (!api || !window)
            return nullptr;

        VkWin32SurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
        createInfo.pNext = nullptr; // pNext must be NULL
        createInfo.flags = static_cast<VkWin32SurfaceCreateFlagsKHR>(0);
        createInfo.hinstance = GetModuleHandle(NULL);
        createInfo.hwnd = static_cast<HWND>(window->getNativeHandle());

        VkSurfaceKHR vk_surface;
        // `vkCreateWin32SurfaceKHR` is taken from `volk` (cause it uses `extern` globals like a n00b)
        if (vkCreateWin32SurfaceKHR(api->getInternalObject(), &createInfo, nullptr, &vk_surface) == VK_SUCCESS)
        {
            auto retval = new this_t(std::move(window), std::move(api), vk_surface);
            return core::smart_refctd_ptr<this_t>(retval, core::dont_grab);
        }
        else
        {
            return nullptr;
        }
    }
};
#elif defined _NBL_PLATFORM_LINUX_
// TODO: later, not this week
#elif defined _NBL_PLATFORM_ANDROID_
// TODO: later, not this week
#endif
}

#endif