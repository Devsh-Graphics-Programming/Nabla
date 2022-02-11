#include <nbl/video/surface/CSurfaceVulkan.h>
#include <nbl/video/CVulkanCommon.h>
#include <nbl/video/CVulkanPhysicalDevice.h>

namespace nbl::video
{
	bool ISurfaceVulkan::isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const
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
	void ISurfaceVulkan::getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const
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
	ISurface::E_PRESENT_MODE ISurfaceVulkan::getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const
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
	bool ISurfaceVulkan::getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const
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
}