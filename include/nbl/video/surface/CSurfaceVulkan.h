#ifndef __NBL_C_SURFACE_VULKAN_H_INCLUDED__
#define __NBL_C_SURFACE_VULKAN_H_INCLUDED__

#include "nbl/video/surface/ISurface.h"

namespace nbl::video
{

class CVulkanConnection;
class CVulkanPhysicalDevice;

template <typename Window>
class CSurfaceVulkan final : public CSurface<Window>
{
public:
    using this_t = CSurfaceVulkan<Window>;
    using base_t = CSurface<Window>;

    static core::smart_refctd_ptr<this_t> create(
        core::smart_refctd_ptr<video::CVulkanConnection>&& api,
        core::smart_refctd_ptr<Window>&& window);

    inline const void* getNativeWindowHandle() const override
    {
        return &base_t::m_window->getNativeHandle();
    }

    bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override;

    inline VkSurfaceKHR getInternalObject() const { return m_vkSurfaceKHR; }

// Todo(achal): Remove
// private:
    explicit CSurfaceVulkan(core::smart_refctd_ptr<video::CVulkanConnection>&& api,
        core::smart_refctd_ptr<Window>&& window, VkSurfaceKHR vk_surface)
        : base_t(std::move(api), std::move(window)), m_vkSurfaceKHR(vk_surface)
    {}

    ~CSurfaceVulkan();

    VkSurfaceKHR m_vkSurfaceKHR = VK_NULL_HANDLE;
};

using CSurfaceVulkanWin32 = CSurfaceVulkan<ui::IWindowWin32>;

}

#endif