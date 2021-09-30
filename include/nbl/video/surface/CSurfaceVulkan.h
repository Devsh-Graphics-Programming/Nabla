#ifndef __NBL_C_SURFACE_VULKAN_H_INCLUDED__
#define __NBL_C_SURFACE_VULKAN_H_INCLUDED__

#include "nbl/video/surface/ISurface.h"
#include "nbl/ui/IWindowAndroid.h"
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

    bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override;

    void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const override;
    
    ISurface::E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const override;
    
    bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const override;

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

#ifdef _NBL_PLATFORM_WINDOWS_
using CSurfaceVulkanWin32 = CSurfaceVulkan<ui::IWindowWin32>;
#elif defined _NBL_PLATFORM_LINUX_
using CSurfaceVulkanX11 = CSurfaceVulkan<ui::IWindowX11>;
#elif defined _NBL_PLATFORM_ANDROID_
using CSurfaceVulkanAndroid = CSurfaceVulkan<ui::IWindowAndroid>;
#endif
}

#endif