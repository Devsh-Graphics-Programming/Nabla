#include <nbl/video/surface/CSurfaceVulkan.h>
#include <nbl/video/CVulkanCommon.h>
#include <nbl/video/CVulkanPhysicalDevice.h>

namespace nbl::video
{
bool ISurfaceVulkan::isSupportedForPhysicalDevice(const IPhysicalDevice* physicalDevice, const uint32_t _queueFamIx) const
{
	if (!physicalDevice || physicalDevice->getAPIType()!=EAT_VULKAN)
		return false;

	VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

	VkBool32 supported;
	if (vkGetPhysicalDeviceSurfaceSupportKHR(vk_physicalDevice,_queueFamIx,m_vkSurfaceKHR,&supported)!=VK_SUCCESS)
		return false;
	return static_cast<bool>(supported);
}

	void ISurfaceVulkan::getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const
	{
		constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;

		if (!physicalDevice || physicalDevice->getAPIType()!=EAT_VULKAN)
			return;

		const VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();

		// the pNext here is for VK_KHR_surface_maintenance1 and fullscreen exclusive surfaces
		const VkPhysicalDeviceSurfaceInfo2KHR vk_surfaceInfo = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, nullptr, m_vkSurfaceKHR };
		VkResult retval = vkGetPhysicalDeviceSurfaceFormats2KHR(vk_physicalDevice, &vk_surfaceInfo, &formatCount, nullptr);

		if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
		{
			formatCount = 0u;
			return;
		}

		if (!formats)
			return;

		VkSurfaceFormat2KHR vk_formats[MAX_SURFACE_FORMAT_COUNT];
		for (auto i=0u; i<formatCount; i++)
			vk_formats[i] = {VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,nullptr};
		retval = vkGetPhysicalDeviceSurfaceFormats2KHR(vk_physicalDevice, &vk_surfaceInfo, &formatCount, vk_formats);

		if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
		{
			formatCount = 0u;
			formats = nullptr;
			return;
		}

		for (uint32_t i = 0u; i < formatCount; ++i)
		{
			formats[i].format = getFormatFromVkFormat(vk_formats[i].surfaceFormat.format);
			formats[i].colorSpace = getColorSpaceFromVkColorSpaceKHR(vk_formats[i].surfaceFormat.colorSpace);
		}
	}

	core::bitflag<ISurface::E_PRESENT_MODE> ISurfaceVulkan::getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const
	{
		constexpr uint32_t MAX_PRESENT_MODE_COUNT = 4u;

		core::bitflag<ISurface::E_PRESENT_MODE> result = ISurface::EPM_UNKNOWN;

		if (!physicalDevice || physicalDevice->getAPIType() != EAT_VULKAN)
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
			result |= getPresentModeFromVkPresentModeKHR(vk_presentModes[i]);

		return result;
	}

	bool ISurfaceVulkan::getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const
	{
		if (!physicalDevice || physicalDevice->getAPIType() != EAT_VULKAN)
			return false;

		const VkPhysicalDevice vk_physicalDevice = static_cast<const CVulkanPhysicalDevice*>(physicalDevice)->getInternalObject();
		const VkPhysicalDeviceSurfaceInfo2KHR vk_surfaceInfo = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, nullptr, m_vkSurfaceKHR };
		// The pNext is for VkDisplayNativeHdrSurfaceCapabilitiesAMD, VkSharedPresentSurfaceCapabilitiesKHR, VkSurfaceCapabilitiesFullScreenExclusiveEXT, VkSurfaceCapabilitiesPresentBarrierNV, VkSurfacePresentModeCompatibilityEXT, VkSurfacePresentScalingCapabilitiesEXT, or VkSurfaceProtectedCapabilitiesKHR
		VkSurfaceCapabilities2KHR vk_surfaceCapabilities = {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,nullptr};
		if (vkGetPhysicalDeviceSurfaceCapabilities2KHR(vk_physicalDevice,&vk_surfaceInfo,&vk_surfaceCapabilities) != VK_SUCCESS)
		{
			return false;
		}

		capabilities.minImageCount = vk_surfaceCapabilities.surfaceCapabilities.minImageCount;
		capabilities.maxImageCount = (vk_surfaceCapabilities.surfaceCapabilities.maxImageCount == 0u) ? ~0u : vk_surfaceCapabilities.surfaceCapabilities.maxImageCount;
		capabilities.currentExtent = vk_surfaceCapabilities.surfaceCapabilities.currentExtent;
		capabilities.minImageExtent = vk_surfaceCapabilities.surfaceCapabilities.minImageExtent;
		capabilities.maxImageExtent = vk_surfaceCapabilities.surfaceCapabilities.maxImageExtent;
		capabilities.maxImageArrayLayers = vk_surfaceCapabilities.surfaceCapabilities.maxImageArrayLayers;
		capabilities.supportedTransforms = static_cast<hlsl::SurfaceTransform::FLAG_BITS>(vk_surfaceCapabilities.surfaceCapabilities.supportedTransforms);
		capabilities.currentTransform = static_cast<hlsl::SurfaceTransform::FLAG_BITS>(vk_surfaceCapabilities.surfaceCapabilities.currentTransform);
		capabilities.supportedCompositeAlpha = static_cast<ISurface::E_COMPOSITE_ALPHA>(vk_surfaceCapabilities.surfaceCapabilities.supportedCompositeAlpha);
		capabilities.supportedUsageFlags = getImageUsageFlagsFromVkImageUsageFlags(vk_surfaceCapabilities.surfaceCapabilities.supportedUsageFlags);

		return true;
	}

ISurfaceVulkan::~ISurfaceVulkan()
{
	vkDestroySurfaceKHR(static_cast<video::CVulkanConnection*>(m_api.get())->getInternalObject(),m_vkSurfaceKHR,nullptr);
}


#ifdef _NBL_PLATFORM_WINDOWS_
core::smart_refctd_ptr<CSurfaceVulkanWin32> CSurfaceVulkanWin32::create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, core::smart_refctd_ptr<ui::IWindowWin32>&& window)
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
	VkResult vkRes = vkCreateWin32SurfaceKHR(api->getInternalObject(), &createInfo, nullptr, &vk_surface);
	if (vkRes != VK_SUCCESS)
		return nullptr;
	auto retval = new this_t(std::move(window), std::move(api), vk_surface);
	return core::smart_refctd_ptr<this_t>(retval, core::dont_grab);
}
core::smart_refctd_ptr<CSurfaceVulkanWin32Native> CSurfaceVulkanWin32Native::create(core::smart_refctd_ptr<video::CVulkanConnection>&& api, ui::IWindowWin32::native_handle_t handle)
{
	if (!api || !handle)
		return nullptr;

	VkWin32SurfaceCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
	createInfo.pNext = nullptr; // pNext must be NULL
	createInfo.flags = static_cast<VkWin32SurfaceCreateFlagsKHR>(0);
	createInfo.hinstance = GetModuleHandle(NULL);
	createInfo.hwnd = static_cast<HWND>(handle);

	VkSurfaceKHR vk_surface;
	// `vkCreateWin32SurfaceKHR` is taken from `volk` (cause it uses `extern` globals like a n00b)
	VkResult vkRes = vkCreateWin32SurfaceKHR(api->getInternalObject(), &createInfo, nullptr, &vk_surface);
	if (vkRes != VK_SUCCESS)
		return nullptr;
	auto retval = new this_t(std::move(api), handle, vk_surface);
	return core::smart_refctd_ptr<this_t>(retval, core::dont_grab);
}
#endif
}