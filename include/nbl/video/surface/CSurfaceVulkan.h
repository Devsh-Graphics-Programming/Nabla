#ifndef _NBL_VIDEO_C_SURFACE_VULKAN_H_INCLUDED_
#define _NBL_VIDEO_C_SURFACE_VULKAN_H_INCLUDED_


#include "BuildConfigOptions.h"

#include "nbl/ui/IWindowAndroid.h"

#include "nbl/video/surface/ISurface.h"
#include "nbl/video/CVulkanConnection.h"

#include <volk/volk.h>

namespace nbl::video
{

class NBL_API2 ISurfaceVulkan : public ISurface
{
        using base_t = ISurface;
    public:
        bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, const uint32_t _queueFamIx) const override;
        void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const override;
        core::bitflag<ISurface::E_PRESENT_MODE> getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const override;
        bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const override;

        inline VkSurfaceKHR getInternalObject() const { return m_vkSurfaceKHR; }

    protected:
        explicit inline ISurfaceVulkan(core::smart_refctd_ptr<video::IAPIConnection>&& api, VkSurfaceKHR vk_surface) : base_t(std::move(api)), m_vkSurfaceKHR(vk_surface) {}
        virtual ~ISurfaceVulkan();

        VkSurfaceKHR m_vkSurfaceKHR = VK_NULL_HANDLE;
};

#ifdef _NBL_PLATFORM_WINDOWS_
class NBL_API2 CSurfaceVulkanWin32 final : public CSurface<ui::IWindowWin32,ISurfaceVulkan>
{
        using this_t = CSurfaceVulkanWin32;
        using base_t = CSurface<ui::IWindowWin32,ISurfaceVulkan>;
    public:
        inline CSurfaceVulkanWin32(core::smart_refctd_ptr<ui::IWindowWin32>&& window, core::smart_refctd_ptr<IAPIConnection>&& api, VkSurfaceKHR surf) :
            base_t(std::move(window), std::move(api), surf) {}
        
        static core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, core::smart_refctd_ptr<ui::IWindowWin32>&& window);
};

class NBL_API2 CSurfaceVulkanWin32Native final : public CSurfaceNative<ui::IWindowWin32, ISurfaceVulkan>
{
        using this_t = CSurfaceVulkanWin32Native;
        using base_t = CSurfaceNative<ui::IWindowWin32, ISurfaceVulkan>;
    public:
        inline CSurfaceVulkanWin32Native(core::smart_refctd_ptr<IAPIConnection>&& api, typename ui::IWindowWin32::native_handle_t handle, VkSurfaceKHR surf) :
            base_t(handle, std::move(api), surf)
        {
        }
        
        static core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, ui::IWindowWin32::native_handle_t handle);
};

#elif defined _NBL_PLATFORM_LINUX_
class NBL_API2 CSurfaceVulkanXcb final : public CSurface<ui::IWindowXcb,ISurfaceVulkan>
{
        using this_t = CSurfaceVulkanXcb;
        using base_t = CSurface<ui::IWindowXcb,ISurfaceVulkan>;
    public:
        inline CSurfaceVulkanXcb(core::smart_refctd_ptr<ui::IWindowXcb>&& window, core::smart_refctd_ptr<IAPIConnection>&& api, VkSurfaceKHR surf) :
            base_t(std::move(window), std::move(api), surf) {}

        static core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, core::smart_refctd_ptr<ui::IWindowXcb>&& window);
};

class NBL_API2 CSurfaceVulkanXcbNative final : public CSurfaceNative<ui::IWindowXcb, ISurfaceVulkan>
{
        using this_t = CSurfaceVulkanXcbNative;
        using base_t = CSurfaceNative<ui::IWindowXcb, ISurfaceVulkan>;
    public:
        inline CSurfaceVulkanXcbNative(core::smart_refctd_ptr<IAPIConnection>&& api, const typename ui::IWindowXcb::native_handle_t& handle, VkSurfaceKHR surf) :
            base_t(handle, std::move(api), surf)
        {
        }

        // the caller keeps the connection open and the window alive for the lifetime of the surface
        static core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, const ui::IWindowXcb::native_handle_t& handle);
};
#elif defined _NBL_PLATFORM_ANDROID_
// TODO: later, not this week
#endif
}

#endif