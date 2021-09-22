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
    VkInstance vk_instance = static_cast<const CVulkanConnection*>(base_t::m_api.get())->getInternalObject();
    vkDestroySurfaceKHR(vk_instance, m_vkSurfaceKHR, nullptr);
}

template <typename Window>
bool CSurfaceVulkan<Window>::isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const
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

template <typename Window>
void CSurfaceVulkan<Window>::getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const
{
    constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;

    if (physicalDevice && physicalDevice->getAPIType() != EAT_VULKAN)
        return;

    VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

    VkResult retval = vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physicalDevice, m_vkSurfaceKHR,
        &formatCount, nullptr);

    // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
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

    // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
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

template <typename Window>
ISurface::E_PRESENT_MODE CSurfaceVulkan<Window>::getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const
{
    constexpr uint32_t MAX_PRESENT_MODE_COUNT = 4u;

    ISurface::E_PRESENT_MODE result = ISurface::EPM_UNKNOWN;

    if (physicalDevice && physicalDevice->getAPIType() != EAT_VULKAN)
        return result;

    VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

    uint32_t count = 0u;
    VkResult retval = vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physicalDevice, m_vkSurfaceKHR,
        &count, nullptr);

    // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
    if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        return result;

    assert(count <= MAX_PRESENT_MODE_COUNT);

    VkPresentModeKHR vk_presentModes[MAX_PRESENT_MODE_COUNT];
    retval = vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physicalDevice, m_vkSurfaceKHR,
        &count, vk_presentModes);

    // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
    if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        return result;

    for (uint32_t i = 0u; i < count; ++i)
        result = static_cast<ISurface::E_PRESENT_MODE>(result | getPresentModeFromVkPresentModeKHR(vk_presentModes[i]));

    return result;
}

template <typename Window>
bool CSurfaceVulkan<Window>::getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const
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
    capabilities.maxImageCount = vk_surfaceCapabilities.maxImageCount;
    capabilities.currentExtent = vk_surfaceCapabilities.currentExtent;
    capabilities.minImageExtent = vk_surfaceCapabilities.minImageExtent;
    capabilities.maxImageExtent = vk_surfaceCapabilities.maxImageExtent;
    capabilities.maxImageArrayLayers = vk_surfaceCapabilities.maxImageArrayLayers;
    // Todo(achal)
    // VkSurfaceTransformFlagsKHR       supportedTransforms;
    // VkSurfaceTransformFlagBitsKHR    currentTransform;
    // VkCompositeAlphaFlagsKHR         supportedCompositeAlpha;
    capabilities.supportedUsageFlags = static_cast<asset::IImage::E_USAGE_FLAGS>(vk_surfaceCapabilities.supportedUsageFlags);

    return true;
}

template class CSurfaceVulkan<ui::IWindowWin32>;

}