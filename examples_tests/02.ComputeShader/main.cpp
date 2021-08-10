#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

// Temporary
#include <volk/volk.h>
#include "../../src/nbl/video/CVulkanPhysicalDevice.h"
#include "../../src/nbl/video/CVKLogicalDevice.h"
#include "../../src/nbl/video/CVulkanImage.h"

#include <nbl/ui/CWindowManagerWin32.h>

using namespace nbl;
using namespace core;

// This probably a TODO for @sadiuk
static bool windowShouldClose_Global = false;

#if 0
const char* src = R"(#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform pushConstants {
    layout (offset = 0) uvec2 imgSize;
} u_pushConstants;

layout (set = 0, binding = 0, rgba8) uniform readonly image2D inImage;
layout (set = 0, binding = 1, rgba8) uniform image2D outImage;

void main()
{
	if (all(lessThan(gl_GlobalInvocationID.xy, u_pushConstants.imgSize)))
	{
		vec3 rgb = imageLoad(inImage, ivec2(gl_GlobalInvocationID.xy)).rgb;
		
		imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), vec4(1, 1, 0, 1));
	}
})";
#else
const char* src = R"(#version 450

// layout (local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0, rgba8) uniform writeonly image2D outImage;

layout (set = 0, binding = 1) uniform UniformBufferObject
{
	float r;
	float g;
	float b;
	float a;
} ubo;

void main()
{
	// if (all(lessThan(gl_GlobalInvocationID.xy, u_pushConstants.imgSize)))
	{
		// vec3 rgb = imageLoad(inImage, ivec2(gl_GlobalInvocationID.xy)).rgb;
		
		// imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), vec4(1, 0, 1, 1));
		imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), vec4(ubo.r, ubo.g, ubo.b, ubo.a));
	}
})";
#endif

inline void debugCallback(nbl::video::E_DEBUG_MESSAGE_SEVERITY severity, nbl::video::E_DEBUG_MESSAGE_TYPE type, const char* msg, void* userData)
{
	using namespace nbl;
	const char* sev = nullptr;
	switch (severity)
	{
	case video::EDMS_VERBOSE:
		sev = "verbose"; break;
	case video::EDMS_INFO:
		sev = "info"; break;
	case video::EDMS_WARNING:
		sev = "warning"; break;
	case video::EDMS_ERROR:
		sev = "error"; break;
	}
	std::cout << "OpenGL " << sev << ": " << msg << std::endl;
}

#define LOG(...) printf(__VA_ARGS__); printf("\n");
class DemoEventCallback : public nbl::ui::IWindow::IEventCallback
{
	bool onWindowShown_impl() override
	{
		LOG("Window Shown");
		return true;
	}
	bool onWindowHidden_impl() override
	{
		LOG("Window hidden");
		return true;
	}
	bool onWindowMoved_impl(int32_t x, int32_t y) override
	{
		LOG("Window window moved to { %d, %d }", x, y);
		return true;
	}
	bool onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		LOG("Window resized to { %u, %u }", w, h);
		return true;
	}
	bool onWindowMinimized_impl() override
	{
		LOG("Window minimized");
		return true;
	}
	bool onWindowMaximized_impl() override
	{
		LOG("Window maximized");
		return true;
	}
	void onGainedMouseFocus_impl() override
	{
		LOG("Window gained mouse focus");
	}
	void onLostMouseFocus_impl() override
	{
		LOG("Window lost mouse focus");
	}
	void onGainedKeyboardFocus_impl() override
	{
		LOG("Window gained keyboard focus");
	}
	void onLostKeyboardFocus_impl() override
	{
		LOG("Window lost keyboard focus");
		windowShouldClose_Global = true;
	}

	void onMouseConnected_impl(core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		LOG("A mouse has been connected");
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		LOG("A mouse has been disconnected");
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		LOG("A keyboard has been connected");
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* mch) override
	{
		LOG("A keyboard has been disconnected");
	}
};

struct alignas(256) UniformBufferObject
{
	float r, g, b, a;
};

struct ArgumentReferenceSegment
{
	std::array<core::smart_refctd_ptr<core::IReferenceCounted>, 63> arguments;

	// What is this nextBlock here for?
	// What emplace would return?
	uint32_t argCount, nextBlock;
};

struct DemoPOD
{
	int32_t x[256];
};

int main()
{
	const size_t BLOCK_SIZE = 4096u * 1024u;
	const size_t MAX_BLOCK_COUNT = 256u;

	// core::CMemoryPool<core::PoolAddressAllocator<uint32_t>,
	// 	core::default_aligned_allocator> mempool(BLOCK_SIZE, MAX_BLOCK_COUNT);

	// core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, core::default_aligned_allocator>::addr_allocator_type::
//	core::address_allocator_traits<core::PoolAddressAllocator<uint32_t>>::multi_alloc_addr()

	// ArgumentReferenceSegment* newSegment = nullptr;
	// UniformBufferObject* newSegment = nullptr;
	// newSegment = mempool.emplace_n<ArgumentReferenceSegment>(1);


	constexpr uint32_t WIN_W = 800u;
	constexpr uint32_t WIN_H = 600u;
	constexpr uint32_t SC_IMG_COUNT = 3u; // problematic, shouldn't fix the number of swapchain images at compile time, since Vulkan is under no obligation to return you the exact number of images you requested

	// Note(achal): This is unused, for now
	video::SDebugCallback dbgcb;
	dbgcb.callback = &debugCallback;
	dbgcb.userData = nullptr;

	auto system = CommonAPI::createSystem(); // Todo(achal): Need to get rid of this

	auto winManager = core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>();

	nbl::ui::IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = WIN_W;
	params.height = WIN_H;
	params.x = 0;
	params.y = 0;
	params.system = core::smart_refctd_ptr(system);
	params.flags = nbl::ui::IWindow::ECF_NONE;
	params.windowCaption = "02.ComputeShader";
	params.callback = core::make_smart_refctd_ptr<DemoEventCallback>();
	auto window = winManager->createWindow(std::move(params));

	core::smart_refctd_ptr<video::IAPIConnection> vk = video::IAPIConnection::create(
		std::move(system), video::EAT_VULKAN, 0, "02.ComputeShader", &dbgcb);
	core::smart_refctd_ptr<video::ISurface> surface = vk->createSurface(window.get());

	// Todo(achal): Remove
	VkSurfaceKHR vk_surface = reinterpret_cast<video::ISurfaceVK*>(surface.get())->m_surface;

	auto gpus = vk->getPhysicalDevices();
	assert(!gpus.empty());

	// I want a GPU which supports both compute queue and present queue
	uint32_t computeFamilyIndex(~0u);
	uint32_t presentFamilyIndex(~0u);

	// Todo(achal): Probably want to put these into some struct
	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode;
	nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform;
	nbl::asset::E_SHARING_MODE imageSharingMode;
	VkExtent2D swapchainExtent;
	using QueueFamilyIndicesArrayType = core::smart_refctd_dynamic_array<uint32_t>;
	QueueFamilyIndicesArrayType queueFamilyIndices;

	// Todo(achal): Should be an IPhysicalDevice
	core::smart_refctd_ptr<video::CVulkanPhysicalDevice> gpu = nullptr;
	for (size_t i = 0ull; i < gpus.size(); ++i)
	{
		// Todo(achal): Hacks, get rid
		gpu = core::smart_refctd_ptr_static_cast<video::CVulkanPhysicalDevice>(*(gpus.begin() + i));

		bool isGPUSuitable = false;

		// Queue families --need to look for compute and present families
		{
			const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin() + familyIndex;
				if (familyProperty->queueFlags & video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_COMPUTE_BIT)
					computeFamilyIndex = familyIndex;

				if (surface->isSupported(gpu.get(), familyIndex))
					presentFamilyIndex = familyIndex;

				if ((computeFamilyIndex != ~0u) && (presentFamilyIndex != ~0u))
				{
					isGPUSuitable = true;
					break;
				}
			}
		}

		// Check if this physical device supports the swapchain extension
		// Todo(achal): Eventually move this to CommonAPI.h
		{
			// Todo(achal): Get this from the user
			const uint32_t requiredDeviceExtensionCount = 1u;
			const char* requiredDeviceExtensionNames[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

			uint32_t availableExtensionCount;
			vkEnumerateDeviceExtensionProperties(gpu->getInternalObject(), NULL, &availableExtensionCount, NULL);
			std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
			vkEnumerateDeviceExtensionProperties(gpu->getInternalObject(), NULL, &availableExtensionCount, availableExtensions.data());

			bool requiredDeviceExtensionsAvailable = false;
			for (uint32_t i = 0u; i < availableExtensionCount; ++i)
			{
				if (strcmp(availableExtensions[i].extensionName, requiredDeviceExtensionNames[0]) == 0)
				{
					requiredDeviceExtensionsAvailable = true;
					break;
				}
			}

			if (!requiredDeviceExtensionsAvailable)
				isGPUSuitable = false;
		}

		// Check if the surface is adequate
		{
			uint32_t surfaceFormatCount;
			gpu->getAvailableFormatsForSurface(surface.get(), surfaceFormatCount, nullptr);
			std::vector<video::ISurface::SFormat> surfaceFormats(surfaceFormatCount);
			gpu->getAvailableFormatsForSurface(surface.get(), surfaceFormatCount, surfaceFormats.data());

			video::ISurface::E_PRESENT_MODE presentModes =
				gpu->getAvailablePresentModesForSurface(surface.get());

			// Todo(achal): Probably should make a ISurface::SCapabilities
			// struct for this as a wrapper for VkSurfaceCapabilitiesKHR
			// nbl::video::ISurface::SCapabilities surfaceCapabilities = ;
			VkSurfaceCapabilitiesKHR surfaceCapabilities;
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu->getInternalObject(),
				vk_surface, &surfaceCapabilities);

			printf("Min swapchain image count: %d\n", surfaceCapabilities.minImageCount);
			printf("Max swapchain image count: %d\n", surfaceCapabilities.maxImageCount);

			assert(surfaceCapabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT);

			if ((surfaceCapabilities.maxImageCount != 0) && (SC_IMG_COUNT > surfaceCapabilities.maxImageCount)
				|| (surfaceFormats.empty()) || (presentModes == static_cast<video::ISurface::E_PRESENT_MODE>(0)))
			{
				isGPUSuitable = false;
			}

			// Todo(achal): Probably a more sophisticated way to choose these
			surfaceFormat = surfaceFormats[0];
			presentMode = static_cast<video::ISurface::E_PRESENT_MODE>(presentModes & (1 << 0));
			preTransform = static_cast<nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS>(surfaceCapabilities.currentTransform);
			swapchainExtent = surfaceCapabilities.currentExtent;
		}

		if (isGPUSuitable) // find the first suitable GPU
			break;
	}
	assert((computeFamilyIndex != ~0u) && (presentFamilyIndex != ~0u));

	video::ILogicalDevice::SCreationParams deviceCreationParams;
	if (computeFamilyIndex == presentFamilyIndex)
	{
		deviceCreationParams.queueParamsCount = 1u;
		imageSharingMode = asset::ESM_EXCLUSIVE;
	}
	else
	{
		deviceCreationParams.queueParamsCount = 2u;
		imageSharingMode = asset::ESM_CONCURRENT;
	}

	queueFamilyIndices = core::make_refctd_dynamic_array<QueueFamilyIndicesArrayType>(deviceCreationParams.queueParamsCount);
	{
		const uint32_t temp[] = { computeFamilyIndex, presentFamilyIndex };
		for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
			(*queueFamilyIndices)[i] = temp[i];
	}

	// Could make this a static member of IGPUQueue, something like
	// IGPUQueue::DEFAULT_QUEUE_PRIORITY, as a "don't-care" value for the user
	const float defaultQueuePriority = 1.f;

	std::vector<video::ILogicalDevice::SQueueCreationParams> queueCreationParams(deviceCreationParams.queueParamsCount);
	for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
	{
		queueCreationParams[i].familyIndex = (*queueFamilyIndices)[i];
		queueCreationParams[i].count = 1u;
		queueCreationParams[i].flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
		queueCreationParams[i].priorities = &defaultQueuePriority;
	}
	deviceCreationParams.queueCreateInfos = queueCreationParams.data();
	core::smart_refctd_ptr<video::ILogicalDevice> device = gpu->createLogicalDevice(std::move(deviceCreationParams));

	video::IGPUQueue* computeQueue = device->getQueue(computeFamilyIndex, 0u);
	video::IGPUQueue* presentQueue = device->getQueue(presentFamilyIndex, 0u);

	// Todo(achal): We might want to check if: VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT and
	// VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT are supported for current surface format
	// and current physical device. In fact, we might want to put this up as a criteria
	// for making the physical device suitable
	nbl::video::ISwapchain::SCreationParams sc_params = {};
	sc_params.surface = surface;
	sc_params.minImageCount = SC_IMG_COUNT;
	sc_params.surfaceFormat = surfaceFormat;
	sc_params.presentMode = presentMode;
	sc_params.width = WIN_W;
	sc_params.height = WIN_H;
	sc_params.queueFamilyIndices = queueFamilyIndices;
	sc_params.imageSharingMode = imageSharingMode;
	sc_params.preTransform = preTransform;
	sc_params.imageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(
		asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);

	core::smart_refctd_ptr<video::ISwapchain> swapchain = device->createSwapchain(
		std::move(sc_params));

	const auto swapchainImages = swapchain->getImages();
	const uint32_t swapchainImageCount = swapchain->getImageCount();
	assert(swapchainImageCount == SC_IMG_COUNT);

	core::smart_refctd_ptr<video::IGPUImageView> swapchainImageViews[SC_IMG_COUNT];
	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		auto img = swapchainImages.begin()[i];
		{
			video::IGPUImageView::SCreationParams viewParams;
			viewParams.format = img->getCreationParameters().format;
			viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.image = std::move(img); // this might create problems

			swapchainImageViews[i] = device->createGPUImageView(std::move(viewParams));
			assert(swapchainImageViews[i]);
		}
	}

	// TODO: Load from "../compute.comp" instead of getting source from src
	core::smart_refctd_ptr<video::IGPUShader> unspecializedShader = device->createGPUShader(
		core::make_smart_refctd_ptr<asset::ICPUShader>(src));
	asset::ISpecializedShader::SInfo specializationInfo(nullptr, nullptr, "main",
		asset::ISpecializedShader::ESS_COMPUTE);
	core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader =
		device->createGPUSpecializedShader(unspecializedShader.get(), specializationInfo);

	core::smart_refctd_ptr<video::IGPUCommandPool> commandPool =
		device->createCommandPool(computeFamilyIndex,
			video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[SC_IMG_COUNT];
	assert(device->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT,
		commandBuffers));

	// Todo(achal): I think we can make it greater than SC_IMG_COUNT
	const uint32_t FRAMES_IN_FLIGHT = 2u;

	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT];
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		acquireSemaphores[i] = device->createSemaphore();
		releaseSemaphores[i] = device->createSemaphore();
		frameFences[i] = device->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
	}
	
	// Hacky vulkan stuff begins --get handles to existing Vulkan stuff
	VkPhysicalDevice vk_physicalDevice = gpu->getInternalObject();
	VkDevice vk_device = reinterpret_cast<video::CVKLogicalDevice*>(device.get())->getInternalObject();

	VkSemaphore vk_acquireSemaphores[FRAMES_IN_FLIGHT], vk_releaseSemaphores[FRAMES_IN_FLIGHT];
	VkFence vk_frameFences[FRAMES_IN_FLIGHT];
	{
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			vk_acquireSemaphores[i] = reinterpret_cast<video::CVulkanSemaphore*>(acquireSemaphores[i].get())->getInternalObject();
			vk_releaseSemaphores[i] = reinterpret_cast<video::CVulkanSemaphore*>(releaseSemaphores[i].get())->getInternalObject();
			vk_frameFences[i] = reinterpret_cast<video::CVulkanFence*>(frameFences[i].get())->getInternalObject();
		}
	}

	VkQueue vk_computeQueue = reinterpret_cast<video::CVulkanQueue*>(
		computeQueue)->getInternalObject();

	VkImage vk_swapchainImages[SC_IMG_COUNT];
	uint32_t i = 0u;
	for (auto image : swapchainImages)
		vk_swapchainImages[i++] = reinterpret_cast<video::CVulkanImage*>(image.get())->getInternalObject();

	VkImageView vk_swapchainImageViews[SC_IMG_COUNT];
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		vk_swapchainImageViews[i] = reinterpret_cast<video::CVulkanImageView*>(
			swapchainImageViews[i].get())->getInternalObject();
	}

	VkShaderModule vk_shaderModule = reinterpret_cast<video::CVulkanSpecializedShader*>(specializedShader.get())->m_shaderModule;

	VkCommandPool vk_commandPool = reinterpret_cast<video::CVulkanCommandPool*>(commandPool.get())->getInternalObject();

	VkCommandBuffer vk_commandBuffers[SC_IMG_COUNT];
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		vk_commandBuffers[i] = reinterpret_cast<video::CVulkanCommandBuffer*>(commandBuffers[i].get())->getInternalObject();

	// Pure vulkan stuff

	// Create UBO (and their memory) per swapchain image
	VkBuffer vk_ubos[SC_IMG_COUNT];
	VkDeviceMemory vk_ubosMemory[SC_IMG_COUNT];
	{
		VkDeviceSize vk_uboSize = sizeof(UniformBufferObject);

		for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		{
			VkBufferCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
			vk_createInfo.pNext = nullptr;
			vk_createInfo.flags = 0;
			vk_createInfo.size = vk_uboSize;
			vk_createInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			vk_createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			vk_createInfo.queueFamilyIndexCount = 0u;
			vk_createInfo.pQueueFamilyIndices = nullptr;

			assert(vkCreateBuffer(vk_device, &vk_createInfo, nullptr, &vk_ubos[i]) == VK_SUCCESS);

			VkMemoryRequirements vk_memoryRequirements;
			vkGetBufferMemoryRequirements(vk_device, vk_ubos[i], &vk_memoryRequirements);

			// Find the type of memory you want to allocate for this buffer, it has to have
			// the following properties:
			// 	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
			// i.e. should be visible to host application and should be coherent with host
			// application

			const uint32_t vk_supportedMemoryTypes = vk_memoryRequirements.memoryTypeBits;
			const VkMemoryPropertyFlags vk_desiredMemoryProperties =
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			uint32_t vk_memoryTypeIndex = ~0u;
			{
				VkPhysicalDeviceMemoryProperties vk_physicalDeviceMemoryProperties;
				vkGetPhysicalDeviceMemoryProperties(vk_physicalDevice,
					&vk_physicalDeviceMemoryProperties);

				for (uint32_t i = 0u; i < vk_physicalDeviceMemoryProperties.memoryTypeCount; ++i)
				{
					const bool isMemoryTypeSupportedForResource =
						(vk_supportedMemoryTypes & (1 << i));

					const bool doesMemoryHaveDesirableProperites =
						(vk_physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags
							& vk_desiredMemoryProperties) == vk_desiredMemoryProperties;
					if (isMemoryTypeSupportedForResource && doesMemoryHaveDesirableProperites)
					{
						vk_memoryTypeIndex = i;
						break;
					}
				}
			}
			assert(vk_memoryTypeIndex != ~0u);

			VkMemoryAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
			vk_allocateInfo.pNext = nullptr;
			vk_allocateInfo.allocationSize = vk_uboSize;
			vk_allocateInfo.memoryTypeIndex = vk_memoryTypeIndex;

			assert(vkAllocateMemory(vk_device, &vk_allocateInfo, nullptr, &vk_ubosMemory[i]) == VK_SUCCESS);

			assert(vkBindBufferMemory(vk_device, vk_ubos[i], vk_ubosMemory[i], 0) == VK_SUCCESS);
		}
	}

	// Todo(achal): Would it make sense to ditch map/unmap now in favor of staging buffer now??
	// Fill up ubos with dummy data
	struct UniformBufferObject uboData_cpu[3] = { { 1.f, 0.f, 0.f, 1.f}, {0.f, 1.f, 0.f, 1.f}, {0.f, 0.f, 1.f, 1.f} };
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		void* mappedMemoryAddress;
		assert(vkMapMemory(vk_device, vk_ubosMemory[i], 0, sizeof(UniformBufferObject), 0,
			&mappedMemoryAddress) == VK_SUCCESS);
		memcpy(mappedMemoryAddress, &uboData_cpu[i], sizeof(UniformBufferObject));
		vkUnmapMemory(vk_device, vk_ubosMemory[i]);
	}

	VkDescriptorSetLayout vk_dsLayout;
	{
		// image2D
		VkDescriptorSetLayoutBinding vk_dsLayoutImageBinding = {};
		vk_dsLayoutImageBinding.binding = 0u;
		vk_dsLayoutImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		vk_dsLayoutImageBinding.descriptorCount = 1u;
		vk_dsLayoutImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		vk_dsLayoutImageBinding.pImmutableSamplers = nullptr;

		// ubo
		VkDescriptorSetLayoutBinding vk_dsLayoutUboBinding = {};
		vk_dsLayoutUboBinding.binding = 1u;
		vk_dsLayoutUboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		vk_dsLayoutUboBinding.descriptorCount = 1u;
		vk_dsLayoutUboBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		vk_dsLayoutUboBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutBinding vk_dsLayoutBindings[] =
		{ vk_dsLayoutImageBinding, vk_dsLayoutUboBinding };

		VkDescriptorSetLayoutCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		createInfo.pNext = nullptr;
		createInfo.flags = 0;
		createInfo.bindingCount = 2u;
		createInfo.pBindings = vk_dsLayoutBindings;

		assert(vkCreateDescriptorSetLayout(vk_device, &createInfo, nullptr, &vk_dsLayout) == VK_SUCCESS);
	}

	VkDescriptorPool vk_descriptorPool = VK_NULL_HANDLE;
	{
		const uint32_t descriptorPoolSizeStructsCount = 2u;
		VkDescriptorPoolSize vk_poolSizes[descriptorPoolSizeStructsCount] = {};
		vk_poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		vk_poolSizes[0].descriptorCount = SC_IMG_COUNT;
		vk_poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		vk_poolSizes[1].descriptorCount = SC_IMG_COUNT;

		VkDescriptorPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		createInfo.pNext = nullptr;
		createInfo.flags = 0;
		createInfo.maxSets = SC_IMG_COUNT;
		createInfo.poolSizeCount = descriptorPoolSizeStructsCount;
		createInfo.pPoolSizes = vk_poolSizes;
		
		assert(vkCreateDescriptorPool(vk_device, &createInfo, nullptr, &vk_descriptorPool) == VK_SUCCESS);
	}

	VkDescriptorSet vk_descriptorSets[SC_IMG_COUNT];
	{
		VkDescriptorSetLayout vk_dsLayouts[SC_IMG_COUNT];
		for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
			vk_dsLayouts[i] = vk_dsLayout;

		VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		allocateInfo.pNext = nullptr;
		allocateInfo.descriptorPool = vk_descriptorPool;
		allocateInfo.descriptorSetCount = SC_IMG_COUNT;
		allocateInfo.pSetLayouts = vk_dsLayouts;

		assert(vkAllocateDescriptorSets(vk_device, &allocateInfo, vk_descriptorSets) == VK_SUCCESS);

		// Update descriptor sets
		for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		{
			VkDescriptorImageInfo vk_imageInfo = {};
			vk_imageInfo.sampler = VK_NULL_HANDLE;
			vk_imageInfo.imageView = vk_swapchainImageViews[i];
			vk_imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			VkDescriptorBufferInfo vk_bufferInfo = {};
			vk_bufferInfo.buffer = vk_ubos[i];
			vk_bufferInfo.offset = 0;
			vk_bufferInfo.range = sizeof(UniformBufferObject);

			const uint32_t descriptorCount = 2u;
			VkWriteDescriptorSet vk_descriptorWrites[descriptorCount] = {};

			// image2D
			vk_descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			vk_descriptorWrites[0].pNext = nullptr;
			vk_descriptorWrites[0].dstSet = vk_descriptorSets[i];
			vk_descriptorWrites[0].dstBinding = 0u;
			vk_descriptorWrites[0].dstArrayElement = 0u;
			vk_descriptorWrites[0].descriptorCount = 1u;
			vk_descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			vk_descriptorWrites[0].pImageInfo = &vk_imageInfo;
			vk_descriptorWrites[0].pBufferInfo = nullptr;
			vk_descriptorWrites[0].pTexelBufferView = nullptr;

			// ubo
			vk_descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			vk_descriptorWrites[1].pNext = nullptr;
			vk_descriptorWrites[1].dstSet = vk_descriptorSets[i];
			vk_descriptorWrites[1].dstBinding = 1u;
			vk_descriptorWrites[1].dstArrayElement = 0u;
			vk_descriptorWrites[1].descriptorCount = 1u; // this is probably for specifying an array of ubos
			vk_descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			vk_descriptorWrites[1].pImageInfo = nullptr;
			vk_descriptorWrites[1].pBufferInfo = &vk_bufferInfo;
			vk_descriptorWrites[1].pTexelBufferView = nullptr;

			vkUpdateDescriptorSets(vk_device, descriptorCount, vk_descriptorWrites, 0u, nullptr);
		}
	}

	VkPipelineLayout vk_pipelineLayout;
	{
		VkPipelineLayoutCreateInfo createInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		createInfo.pNext = nullptr;
		createInfo.flags = 0;
		createInfo.setLayoutCount = 1u;
		createInfo.pSetLayouts = &vk_dsLayout;
		createInfo.pushConstantRangeCount = 0u;
		createInfo.pPushConstantRanges = nullptr;

		assert(vkCreatePipelineLayout(vk_device, &createInfo, nullptr, &vk_pipelineLayout) == VK_SUCCESS);
	}

	VkPipeline vk_pipeline;
	{
		VkPipelineShaderStageCreateInfo vk_shaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		vk_shaderStageCreateInfo.pNext = nullptr;
		vk_shaderStageCreateInfo.flags = 0;
		vk_shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		vk_shaderStageCreateInfo.module = vk_shaderModule;
		vk_shaderStageCreateInfo.pName = "main"; // Todo(achal): Get from shader specialization info
		vk_shaderStageCreateInfo.pSpecializationInfo = nullptr;

		VkComputePipelineCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		createInfo.pNext = nullptr;
		createInfo.flags = 0;
		createInfo.stage = vk_shaderStageCreateInfo;
		createInfo.layout = vk_pipelineLayout;
		createInfo.basePipelineHandle = VK_NULL_HANDLE;
		createInfo.basePipelineIndex = -1;

		assert(vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1u, &createInfo,
			nullptr, &vk_pipeline) == VK_SUCCESS);
	}

	// Record commands in commandBuffers here
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		VkImageMemoryBarrier undefToComputeBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		undefToComputeBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		undefToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		undefToComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		undefToComputeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		undefToComputeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		undefToComputeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		undefToComputeBarrier.image = vk_swapchainImages[i];

		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.levelCount = 1u;
		subresourceRange.layerCount = 1u;
		undefToComputeBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier computeToPresentBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		computeToPresentBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		computeToPresentBarrier.dstAccessMask = 0;
		computeToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		computeToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		computeToPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		computeToPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		computeToPresentBarrier.image = vk_swapchainImages[i];
		computeToPresentBarrier.subresourceRange = subresourceRange;

		commandBuffers[i]->begin(0);

		// The fact that this pipeline barrier is solely on a compute queue might
		// affect the srcStageMask. More precisely, I think, for some reason, that
		// VK_PIPELINE_STAGE_TRANSFER_BIT shouldn't be specified in compute queue
		// but present queue (or transfer queue if theres one??)
		vkCmdPipelineBarrier(vk_commandBuffers[i], VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0u, nullptr, 0u, nullptr, 1u, &undefToComputeBarrier);

		video::IGPUCommandBuffer::SImageMemoryBarrier undefToComputeTransitionBarrier;
		// asset::SMemoryBarrier barrier;
		// asset::E_IMAGE_LAYOUT oldLayout;
		// asset::E_IMAGE_LAYOUT newLayout;
		// uint32_t srcQueueFamilyIndex;
		// uint32_t dstQueueFamilyIndex;
		// core::smart_refctd_ptr<const image_t> image;
		// asset::IImage::SSubresourceRange subresourceRange;
		// undefToComputeTransitionBarrier.
		commandBuffers[i]->pipelineBarrier(asset::EPSF_TRANSFER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT, 0, 0u, nullptr, 0u, nullptr, 1u,
			&undefToComputeTransitionBarrier);

		vkCmdBindPipeline(vk_commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);

		vkCmdBindDescriptorSets(vk_commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE,
			vk_pipelineLayout, 0u, 1u, &vk_descriptorSets[i], 0u, nullptr);

		vkCmdDispatch(vk_commandBuffers[i], WIN_W, WIN_H, 1);

		vkCmdPipelineBarrier(vk_commandBuffers[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0u, nullptr, 0u, nullptr, 1u, &computeToPresentBarrier);

		commandBuffers[i]->end();
	}

	video::ISwapchain* rawPointerToSwapchain = swapchain.get();
	
	uint32_t currentFrameIndex = 0u;
	while (!windowShouldClose_Global)
	{
		video::IGPUSemaphore* acquireSemaphore_frame = acquireSemaphores[currentFrameIndex].get();
		video::IGPUSemaphore* releaseSemaphore_frame = releaseSemaphores[currentFrameIndex].get();
		video::IGPUFence* fence_frame = frameFences[currentFrameIndex].get();

		assert(device->waitForFences(1u, &fence_frame, true, ~0ull) == video::IGPUFence::ES_SUCCESS);

		uint32_t imageIndex;
		swapchain->acquireNextImage(~0ull, acquireSemaphores[currentFrameIndex].get(), nullptr,
			&imageIndex);

		// At this stage the final color values are output from the pipeline
		// Todo(achal): Not really sure why are waiting at this pipeline stage for
		// acquiring the image to render
		VkPipelineStageFlags pipelineStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.waitSemaphoreCount = 1u;
		submitInfo.pWaitSemaphores = &vk_acquireSemaphores[currentFrameIndex];
		submitInfo.pWaitDstStageMask = &pipelineStageFlags;
		submitInfo.commandBufferCount = 1u;
		submitInfo.pCommandBuffers = &vk_commandBuffers[imageIndex];
		submitInfo.signalSemaphoreCount = 1u;
		submitInfo.pSignalSemaphores = &vk_releaseSemaphores[currentFrameIndex];

		// Make sure you unsignal the fence before expecting vkQueueSubmit to signal it
		// once it finishes its execution
		device->resetFences(1u, &fence_frame);

		VkResult result = vkQueueSubmit(vk_computeQueue, 1u, &submitInfo, vk_frameFences[currentFrameIndex]);
		assert(result == VK_SUCCESS);

		// asset::E_PIPELINE_STAGE_FLAGS waitDstStageFlags = asset::E_PIPELINE_STAGE_FLAGS::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;

		// video::IGPUQueue::SSubmitInfo submitInfo = {};
		// submitInfo.waitSemaphoreCount = 1u;
		// submitInfo.pWaitSemaphores = &acquireSemaphore_frame;
		// submitInfo.pWaitDstStageMask = &waitDstStageFlags;
		// submitInfo.signalSemaphoreCount = 1u;
		// submitInfo.pSignalSemaphores = &releaseSemaphore_frame;
		// submitInfo.commandBufferCount = 1u;
		// submitInfo.commandBuffers = ;
		// assert(graphicsQueue->submit(1u, &submitInfo, fence_frame));

		video::IGPUQueue::SPresentInfo presentInfo;
		presentInfo.waitSemaphoreCount = 1u;
		presentInfo.waitSemaphores = &releaseSemaphore_frame;
		presentInfo.swapchainCount = 1u;
		presentInfo.swapchains = &rawPointerToSwapchain;
		presentInfo.imgIndices = &imageIndex;
		assert(presentQueue->present(presentInfo));

		currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
	}
	device->waitIdle();

	vkDestroyCommandPool(vk_device, vk_commandPool, nullptr);

#if 0
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "Compute Shader");
	auto win = initOutp.window;
	auto gl = initOutp.apiConnection;
	auto surface = initOutp.surface;
	auto device = initOutp.logicalDevice;
	auto queue = initOutp.queue;
	auto sc = initOutp.swapchain;
	auto renderpass = initOutp.renderpass;
	auto fbo = initOutp.fbo;
	auto cmdpool = initOutp.commandPool;

	core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool;
	{
		video::IDescriptorPool::E_CREATE_FLAGS flags = video::IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT;
		video::IDescriptorPool::SDescriptorPoolSize poolSize{ nbl::asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_IMAGE, 2 };

		descriptorPool = device->createDescriptorPool(flags, 1, 1, &poolSize);
	}

	//TODO: Load inImgPair from "../../media/color_space_test/R8G8B8A8_2.png" instead of creating empty GPU IMAGE
	auto inImgPair = CommonAPI::createEmpty2DTexture(device, WIN_W, WIN_H);
	auto outImgPair = CommonAPI::createEmpty2DTexture(device, WIN_W, WIN_H);

	core::smart_refctd_ptr<video::IGPUImage> inImg = inImgPair.first;
	core::smart_refctd_ptr<video::IGPUImage> outImg = outImgPair.first;
	core::smart_refctd_ptr<video::IGPUImageView> inImgView = inImgPair.second;
	core::smart_refctd_ptr<video::IGPUImageView> outImgView = outImgPair.second;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> ds0layout;
	{
		video::IGPUDescriptorSetLayout::SBinding bnd[2];
		bnd[0].binding = 0u;
		bnd[0].type = asset::EDT_STORAGE_IMAGE;
		bnd[0].count = 1u;
		bnd[0].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bnd[0].samplers = nullptr;
		bnd[1] = bnd[0];
		bnd[1].binding = 1u;
		ds0layout = device->createGPUDescriptorSetLayout(bnd, bnd + 2);
	}

	core::smart_refctd_ptr<video::IGPUDescriptorSet> ds0_gpu;
	ds0_gpu = device->createGPUDescriptorSet(descriptorPool.get(), ds0layout);
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].arrayElement = 0u;
		write[0].binding = 0u;
		write[0].count = 1u;
		write[0].descriptorType = asset::EDT_STORAGE_IMAGE;
		write[0].dstSet = ds0_gpu.get();
		info[0].desc = inImgView;
		info[0].image.imageLayout = asset::EIL_GENERAL;
		write[0].info = info;
		write[1] = write[0];
		write[1].binding = 1u;
		info[1].desc = outImgView;
		info[1].image.imageLayout = asset::EIL_GENERAL;
		write[1].info = info+1;
		device->updateDescriptorSets(2u, write, 0u, nullptr);
	}

	core::smart_refctd_ptr<video::IGPUComputePipeline> compPipeline;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
	{
		{
			asset::SPushConstantRange range;
			range.offset = 0u;
			range.size = sizeof(uint32_t) * 2u;
			range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
			layout = device->createGPUPipelineLayout(&range, &range + 1, std::move(ds0layout));
		}
		core::smart_refctd_ptr<video::IGPUSpecializedShader> shader;
		{
			//TODO: Load from "../compute.comp" instead of getting source from src
			auto cs_unspec = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(src));
			asset::ISpecializedShader::SInfo csinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE, "cs");
			auto cs = device->createGPUSpecializedShader(cs_unspec.get(), csinfo);

			compPipeline = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(layout), std::move(cs));
		}
	}


	{
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
		assert(cb);

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = WIN_W;
		vp.height = WIN_H;
		cb->setViewport(0u, 1u, &vp);
		cb->end();

		video::IGPUQueue::SSubmitInfo info;
		auto* cb_ = cb.get();
		info.commandBufferCount = 1u;
		info.commandBuffers = &cb_;
		info.pSignalSemaphores = nullptr;
		info.signalSemaphoreCount = 0u;
		info.pWaitSemaphores = nullptr;
		info.waitSemaphoreCount = 0u;
		info.pWaitDstStageMask = nullptr;
		queue->submit(1u, &info, nullptr);
	}

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
	device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
	auto sc_images = sc->getImages();
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = cmdbuf[i];
		auto& fb = fbo[i];

		asset::IImage::SImageCopy region;
		region.dstOffset = { 0, 0, 0 };
		region.srcOffset = { 0, 0, 0 };
		region.extent = { WIN_W, WIN_H, 1 };
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.mipLevel = 0;
		region.dstSubresource.layerCount = 1;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.mipLevel = 0;
		region.srcSubresource.layerCount = 1;
		cb->begin(0);
		cb->bindDescriptorSets(nbl::asset::E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout.get(), 0, 1, (nbl::video::IGPUDescriptorSet**)&ds0_gpu.get());
		cb->pushConstants(layout.get(), asset::ISpecializedShader::ESS_COMPUTE, 0, sizeof(uint32_t) * 2u, &core::vector2du32_SIMD(WIN_W, WIN_H));
		cb->bindComputePipeline(compPipeline.get());
		cb->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);
		video::IGPUCommandBuffer::SImageMemoryBarrier b;
		b.dstQueueFamilyIndex = 0;
		b.srcQueueFamilyIndex = 0;
		b.image = outImg;
		b.newLayout = asset::EIL_UNDEFINED;
		b.oldLayout = asset::EIL_UNDEFINED;
		b.subresourceRange.baseArrayLayer = 0;
		b.subresourceRange.baseMipLevel = 0;
		b.subresourceRange.layerCount = 1;
		b.subresourceRange.levelCount = 1;
		b.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		b.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
		cb->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_TRANSFER_BIT, 0, 0u, nullptr, 0u, nullptr, 1, &b);
		cb->copyImage(outImg.get(), nbl::asset::E_IMAGE_LAYOUT::EIL_UNDEFINED, sc_images.begin()[i].get(), nbl::asset::E_IMAGE_LAYOUT::EIL_UNDEFINED, 1, &region);
		
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		asset::VkRect2D area;
		region.srcOffset = { 0, 0, 0 };
		area.offset = { 0, 0 };
		area.extent = { WIN_W, WIN_H };
		clear.color.float32[0] = 1.f;
		clear.color.float32[1] = 0.f;
		clear.color.float32[2] = 0.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = renderpass;
		info.framebuffer = fb;
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		//cb->beginRenderPass(&info, asset::ESC_INLINE);
		//cb->endRenderPass();

		cb->end();
	}


	constexpr uint32_t FRAME_COUNT = 50000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		auto img_acq_sem = device->createSemaphore();
		auto render1_finished_sem = device->createSemaphore();

		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		CommonAPI::Submit(device.get(), sc.get(), cmdbuf, queue, img_acq_sem.get(), render1_finished_sem.get(), SC_IMG_COUNT, imgnum);

		CommonAPI::Present(device.get(), sc.get(), queue, render1_finished_sem.get(), imgnum);
	}

	device->waitIdle();

#endif
	// return 0;
	exit(0);
}
