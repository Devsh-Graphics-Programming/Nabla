#ifndef __NBL_C_SURFACE_VULKAN_H_INCLUDED__
#define __NBL_C_SURFACE_VULKAN_H_INCLUDED__

#include "nbl/video/surface/ISurface.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanConnection.h"

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

    static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, core::smart_refctd_ptr<Window>&& window)
    {
        if (!api || !window)
            return nullptr;

        #ifdef _NBL_PLATFORM_WINDOWS_    
            // This needs to know what ui::IWindowWin32 is! Won't work on other platforms!
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
                    auto retval = new this_t(std::move(api), std::move(window), vk_surface);
                    return core::smart_refctd_ptr<this_t>(retval, core::dont_grab);
                }
                else
                {
                    return nullptr;
                }
            }
        #else
        return nullptr;
        #endif
    }

    inline bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override
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

    inline void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const override
    {
        constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;

        if (physicalDevice && physicalDevice->getAPIType() != EAT_VULKAN)
            return;

        VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

        VkResult retval = vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physicalDevice, m_vkSurfaceKHR,
            &formatCount, nullptr);

        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        {
            formatCount = 0u;
            return;
        }

        if (!formats)
            return;

        VkSurfaceFormatKHR vk_formats[MAX_SURFACE_FORMAT_COUNT];
        retval = vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physicalDevice, m_vkSurfaceKHR,
            &formatCount, vk_formats);

        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        {
            formatCount = 0u;
            formats = nullptr;
            return;
        }

        for (uint32_t i = 0u; i < formatCount; ++i)
        {
            formats[i].format = getFormatFromVkFormat(vk_formats[i].format);
            formats[i].colorSpace = getColorSpaceFromVkColorSpaceKHR(vk_formats[i].colorSpace);
        }
    }

    inline ISurface::E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const override
    {
        constexpr uint32_t MAX_PRESENT_MODE_COUNT = 4u;

        ISurface::E_PRESENT_MODE result = ISurface::EPM_UNKNOWN;

        if (physicalDevice && physicalDevice->getAPIType() != EAT_VULKAN)
            return result;

        VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

        uint32_t count = 0u;
        VkResult retval = vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physicalDevice, m_vkSurfaceKHR,
            &count, nullptr);

        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return result;

        assert(count <= MAX_PRESENT_MODE_COUNT);

        VkPresentModeKHR vk_presentModes[MAX_PRESENT_MODE_COUNT];
        retval = vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physicalDevice, m_vkSurfaceKHR,
            &count, vk_presentModes);

        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return result;

        for (uint32_t i = 0u; i < count; ++i)
            result = static_cast<ISurface::E_PRESENT_MODE>(result | getPresentModeFromVkPresentModeKHR(vk_presentModes[i]));

        return result;
    }

    inline bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const override
    {
        if (physicalDevice && physicalDevice->getAPIType() != EAT_VULKAN)
            return false;

        VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

        VkSurfaceCapabilitiesKHR vk_surfaceCapabilities;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_physicalDevice, m_vkSurfaceKHR,
            &vk_surfaceCapabilities) != VK_SUCCESS)
        {
            return false;
        }

        capabilities.minImageCount = vk_surfaceCapabilities.minImageCount;
        capabilities.maxImageCount = (vk_surfaceCapabilities.maxImageCount == 0u) ? ~0u : vk_surfaceCapabilities.maxImageCount;
        capabilities.currentExtent = vk_surfaceCapabilities.currentExtent;
        capabilities.minImageExtent = vk_surfaceCapabilities.minImageExtent;
        capabilities.maxImageExtent = vk_surfaceCapabilities.maxImageExtent;
        capabilities.maxImageArrayLayers = vk_surfaceCapabilities.maxImageArrayLayers;
        capabilities.supportedTransforms = static_cast<ISurface::E_SURFACE_TRANSFORM_FLAGS>(vk_surfaceCapabilities.supportedTransforms);
        capabilities.currentTransform = static_cast<ISurface::E_SURFACE_TRANSFORM_FLAGS>(vk_surfaceCapabilities.currentTransform);
        capabilities.supportedCompositeAlpha = static_cast<ISurface::E_COMPOSITE_ALPHA>(vk_surfaceCapabilities.supportedCompositeAlpha);
        capabilities.supportedUsageFlags = static_cast<asset::IImage::E_USAGE_FLAGS>(vk_surfaceCapabilities.supportedUsageFlags);

        return true;
    }

    inline VkSurfaceKHR getInternalObject() const { return m_vkSurfaceKHR; }

protected:
    explicit CSurfaceVulkan(core::smart_refctd_ptr<video::CVulkanConnection>&& api,
        core::smart_refctd_ptr<Window>&& window, VkSurfaceKHR vk_surface)
        : base_t(std::move(api), std::move(window)), m_vkSurfaceKHR(vk_surface)
    {}

    inline ~CSurfaceVulkan()
    {
        VkInstance vk_instance = static_cast<const CVulkanConnection*>(base_t::m_api.get())->getInternalObject();
        vkDestroySurfaceKHR(vk_instance, m_vkSurfaceKHR, nullptr);
    }

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