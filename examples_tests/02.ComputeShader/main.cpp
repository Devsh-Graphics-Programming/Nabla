#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

// Temporary
#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"
#include "../../src/nbl/video/CVulkanConnection.h"
#include "../../src/nbl/video/CVulkanCommon.h"

#include <nbl/ui/CWindowManagerWin32.h>

using namespace nbl;

const char* src = R"(#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (push_constant) uniform pushConstants
{
	layout (offset = 0) uvec2 imgSize;
} u_pushConstants;

layout (set = 0, binding = 0, rgba8) uniform writeonly image2D outImage;
layout (set = 0, binding = 1, rgba8) uniform readonly image2D inImage;

void main()
{
	if (all(lessThan(gl_GlobalInvocationID.xy, u_pushConstants.imgSize)))
	{
		vec3 inColor = imageLoad(inImage, ivec2(gl_GlobalInvocationID.xy)).rgb;
		float grayscale = 0.2126 * inColor.r + 0.7152 * inColor.g + 0.0722 * inColor.b;
		
		imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), vec4(vec3(grayscale), 1.f));
	}
})";

int main()
{
	constexpr uint32_t WIN_W = 768;
	constexpr uint32_t WIN_H = 512u;
	constexpr uint32_t MAX_SWAPCHAIN_IMAGE_COUNT = 8u;
	constexpr uint32_t FBO_COUNT = 2u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);

	auto system = CommonAPI::createSystem();
	auto logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>();
	auto inputSystem = core::make_smart_refctd_ptr<CommonAPI::InputSystem>(system::logger_opt_smart_ptr(logger));
	auto eventCallback = core::make_smart_refctd_ptr<CommonAPI::CommonAPIEventCallback>(core::smart_refctd_ptr(inputSystem), system::logger_opt_smart_ptr(logger));
	auto winManager = core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>();

	nbl::ui::IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = WIN_W;
	params.height = WIN_H;
	params.x = 100;
	params.y = 100;
	params.system = core::smart_refctd_ptr(system);
	params.flags = nbl::ui::IWindow::ECF_NONE;
	params.windowCaption = "02.ComputeShader";
	params.callback = eventCallback;
	auto window = winManager->createWindow(std::move(params));

	auto assetManager = core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(nbl::core::smart_refctd_ptr(system));

	core::smart_refctd_ptr<video::CVulkanConnection> apiConnection =
		video::CVulkanConnection::create(core::smart_refctd_ptr(system), 0, "02.ComputeShader", true);

	core::smart_refctd_ptr<video::CSurfaceVulkanWin32> surface =
		video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(apiConnection),
			core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));

#if 0
	// Todo(achal): Pending bug investigation
	{
		auto opengl_logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>();
		core::smart_refctd_ptr<video::COpenGLConnection> opengl =
			video::COpenGLConnection::create(core::smart_refctd_ptr(system), 0, "02.ComputeShader", video::COpenGLDebugCallback(core::smart_refctd_ptr(opengl_logger)));

		core::smart_refctd_ptr<video::CSurfaceGLWin32> surface_opengl =
			video::CSurfaceGLWin32::create(core::smart_refctd_ptr(opengl),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
	}
#endif

	auto gpus = apiConnection->getPhysicalDevices();
	assert(!gpus.empty());

	// I want a GPU which supports both compute queue and present queue
	uint32_t computeFamilyIndex(~0u);
	uint32_t presentFamilyIndex(~0u);

	// Todo(achal): Probably want to put these into some struct
	uint32_t minSwapchainImageCount(~0u);
	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode;
	// nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform; // Todo(achal)
	nbl::asset::E_SHARING_MODE imageSharingMode;
	VkExtent2D swapchainExtent;

	// Todo(achal): Abstract this out
	video::IPhysicalDevice* gpu = nullptr;
	for (size_t i = 0ull; i < gpus.size(); ++i)
	{
		gpu = gpus.begin()[i];

		bool isGPUSuitable = false;

		// Todo(achal): Abstract out
		// Queue families --need to look for compute and present families
		{
			const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin() + familyIndex;

				if (familyProperty->queueFlags.value & video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_COMPUTE_BIT)
					computeFamilyIndex = familyIndex;

				if (surface->isSupportedForPhysicalDevice(gpu, familyIndex))
					presentFamilyIndex = familyIndex;

				if ((computeFamilyIndex != ~0u) && (presentFamilyIndex != ~0u))
				{
					isGPUSuitable = true;
					break;
				}
			}
		}

		// Since our workload is not headless compute, a swapchain is mandatory
		if (!gpu->isSwapchainSupported())
			isGPUSuitable = false;

		// Todo(achal): Abstract it out
		// Check if the surface is adequate
		{
			uint32_t surfaceFormatCount;
			surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, nullptr);
			std::vector<video::ISurface::SFormat> surfaceFormats(surfaceFormatCount);
			surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, surfaceFormats.data());

			video::ISurface::E_PRESENT_MODE availablePresentModes =
				surface->getAvailablePresentModesForPhysicalDevice(gpu);

			video::ISurface::SCapabilities surfaceCapabilities = {};
			if (!surface->getSurfaceCapabilitiesForPhysicalDevice(gpu, surfaceCapabilities))
				isGPUSuitable = false;

			printf("Min swapchain image count: %d\n", surfaceCapabilities.minImageCount);
			printf("Max swapchain image count: %d\n", surfaceCapabilities.maxImageCount);

			// This is probably required because we're using swapchain image as storage image
			// in this example
			if ((surfaceCapabilities.supportedUsageFlags & asset::IImage::EUF_STORAGE_BIT) == 0)
				isGPUSuitable = false;
			
			if ((surfaceFormats.empty()) || (availablePresentModes == video::ISurface::EPM_UNKNOWN))
				isGPUSuitable = false;

			// Todo(achal): Probably a more sophisticated way to choose these
			minSwapchainImageCount = core::min(surfaceCapabilities.minImageCount + 1u, MAX_SWAPCHAIN_IMAGE_COUNT);
			if ((surfaceCapabilities.maxImageCount != 0u) && (minSwapchainImageCount > surfaceCapabilities.maxImageCount))
				minSwapchainImageCount = surfaceCapabilities.maxImageCount;

			surfaceFormat = surfaceFormats[0];
			presentMode = static_cast<video::ISurface::E_PRESENT_MODE>(availablePresentModes & (1 << 0));
			// preTransform = static_cast<nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS>(surfaceCapabilities.currentTransform);
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

	std::vector<uint32_t> queueFamilyIndices(deviceCreationParams.queueParamsCount);
	{
		const uint32_t tmp[] = { computeFamilyIndex, presentFamilyIndex };
		for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
			queueFamilyIndices[i] = tmp[i];
	}

	const float defaultQueuePriority = video::IGPUQueue::DEFAULT_QUEUE_PRIORITY;

	std::vector<video::ILogicalDevice::SQueueCreationParams> queueCreationParams(deviceCreationParams.queueParamsCount);
	for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
	{
		queueCreationParams[i].familyIndex = queueFamilyIndices[i];
		queueCreationParams[i].count = 1u;
		queueCreationParams[i].flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
		queueCreationParams[i].priorities = &defaultQueuePriority;
	}
	deviceCreationParams.queueParams = queueCreationParams.data();

	core::smart_refctd_ptr<video::ILogicalDevice> device = gpu->createLogicalDevice(deviceCreationParams);

	video::IGPUQueue* computeQueue = device->getQueue(computeFamilyIndex, 0u);
	video::IGPUQueue* presentQueue = device->getQueue(presentFamilyIndex, 0u);

	nbl::video::ISwapchain::SCreationParams scParams = {};
	scParams.surface = surface;
	scParams.minImageCount = minSwapchainImageCount;
	scParams.surfaceFormat = surfaceFormat;
	scParams.presentMode = presentMode;
	scParams.width = WIN_W;
	scParams.height = WIN_H;
	scParams.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
	scParams.queueFamilyIndices = queueFamilyIndices.data();
	scParams.imageSharingMode = imageSharingMode;
	scParams.imageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(
		asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
	scParams.oldSwapchain = nullptr;

	core::smart_refctd_ptr<video::ISwapchain> swapchain = device->createSwapchain(
		std::move(scParams));

	const auto swapchainImages = swapchain->getImages();
	const uint32_t swapchainImageCount = swapchain->getImageCount();

	core::smart_refctd_ptr<video::IGPUImageView> swapchainImageViews[MAX_SWAPCHAIN_IMAGE_COUNT];
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
			viewParams.image = core::smart_refctd_ptr<video::IGPUImage>(img);

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

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[MAX_SWAPCHAIN_IMAGE_COUNT];
	device->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY,
		swapchainImageCount, commandBuffers);

	const uint32_t bindingCount = 2u;
	video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
	{
		// image2D
		bindings[0].binding = 0u;
		bindings[0].type = asset::EDT_STORAGE_IMAGE;
		bindings[0].count = 1u;
		bindings[0].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bindings[0].samplers = nullptr;

		// ubo
		bindings[1].binding = 1u;
		bindings[1].type = asset::EDT_STORAGE_IMAGE;
		bindings[1].count = 1u;
		bindings[1].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bindings[1].samplers = nullptr;
	}
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout =
		device->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

	const uint32_t descriptorPoolSizeCount = 1u;
	video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
	poolSizes[0].type = asset::EDT_STORAGE_IMAGE;
	poolSizes[0].count = swapchainImageCount + 1u;

	video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
		static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

	core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
		= device->createDescriptorPool(descriptorPoolFlags, swapchainImageCount,
			descriptorPoolSizeCount, poolSizes);

	// For each swapchain image we have one descriptor set with two descriptors each
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets[MAX_SWAPCHAIN_IMAGE_COUNT];

	// Todo(achal): Test this as well: 
	// device->createGPUDescriptorSets(descriptorPool.get(), SC_IMG_COUNT, )
	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		descriptorSets[i] = device->createGPUDescriptorSet(descriptorPool.get(),
			core::smart_refctd_ptr(dsLayout));
	}

	// Uncomment once the KTX loader works
#if 0
	constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(
		asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

	const char* pathToImage = "../../media/color_space_test/kueken7_rgba8_unorm.ktx";
	
	asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
	auto cpuImageBundle = assetManager->getAsset(pathToImage, loadParams);
	auto cpuImageContents = cpuImageBundle.getContents();
	if (cpuImageContents.empty())
	{
		logger->log("Failed to read image at path %s", nbl::system::ILogger::ELL_ERROR, pathToImage);
		exit(-1);
	}

	auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuImageContents.begin());
#else
	const uint32_t imageWidth = WIN_W;
	const uint32_t imageHeight = WIN_H;
	const uint32_t imageChannelCount = 4u;
	const uint32_t mipLevels = 1u; // WILL NOT WORK FOR MORE THAN 1 MIPS
	const size_t imageSize = imageWidth * imageHeight * imageChannelCount * sizeof(uint8_t);
	auto imagePixels = core::make_smart_refctd_ptr<asset::ICPUBuffer>(imageSize);

	uint32_t* dstPixel = (uint32_t*)imagePixels->getPointer();
	for (uint32_t y = 0u; y < imageHeight; ++y)
	{
		for (uint32_t x = 0u; x < imageWidth; ++x)
		{
			// Should be red in R8G8B8A8_UNORM
			*dstPixel++ = 0x000000FF;
		}
	}

	core::smart_refctd_ptr<asset::ICPUImage> inImage_CPU = nullptr;
	{
		asset::ICPUImage::SCreationParams creationParams = {};
		creationParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		creationParams.type = asset::IImage::ET_2D;
		creationParams.format = asset::EF_R8G8B8A8_UNORM;
		creationParams.extent = { imageWidth, imageHeight, 1u };
		creationParams.mipLevels = mipLevels;
		creationParams.arrayLayers = 1u;
		creationParams.samples = asset::IImage::ESCF_1_BIT;
		creationParams.tiling = asset::IImage::ET_OPTIMAL;
		{
			// Todo(achal): Need to wrap this up
			VkFormatProperties formatProps;
			vkGetPhysicalDeviceFormatProperties(
				static_cast<video::CVulkanPhysicalDevice*>(gpu)->getInternalObject(),
				video::getVkFormatFromFormat(creationParams.format), &formatProps);
			assert(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
			assert(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT); // this is required to use vkCmdBlitImage to generate mipmaps, this check should be included in asset converter after we have other ways for mip map generation
		}
		creationParams.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_DST_BIT;
		creationParams.sharingMode = asset::ESM_EXCLUSIVE;
		creationParams.queueFamilyIndexCount = 1u;
		creationParams.queueFamilyIndices = nullptr;
		creationParams.initialLayout = asset::EIL_UNDEFINED;

		auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1ull);
		imageRegions->begin()->bufferOffset = 0ull;
		imageRegions->begin()->bufferRowLength = creationParams.extent.width;
		imageRegions->begin()->bufferImageHeight = 0u;
		imageRegions->begin()->imageSubresource = {};
		imageRegions->begin()->imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
		imageRegions->begin()->imageSubresource.layerCount = 1u;
		imageRegions->begin()->imageOffset = { 0, 0, 0 };
		imageRegions->begin()->imageExtent = { creationParams.extent.width, creationParams.extent.height, 1u };

		inImage_CPU = asset::ICPUImage::create(std::move(creationParams));
		inImage_CPU->setBufferAndRegions(core::smart_refctd_ptr<asset::ICPUBuffer>(imagePixels), imageRegions);
	}
#endif

	core::smart_refctd_ptr<video::IUtilities> utils = core::make_smart_refctd_ptr<video::IUtilities>(core::smart_refctd_ptr<video::ILogicalDevice>(device));

	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams = {};
	cpu2gpuParams.utilities = utils.get();
	cpu2gpuParams.device = device.get();
	cpu2gpuParams.assetManager = assetManager.get();
	cpu2gpuParams.pipelineCache = nullptr;
	cpu2gpuParams.limits = gpu->getLimits();
	cpu2gpuParams.finalQueueFamIx = 0u; // queue at index 0 supports both compute and present for me
	cpu2gpuParams.sharingMode = asset::ESM_EXCLUSIVE;
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = computeQueue;
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = computeQueue;

	video::IGPUObjectFromAssetConverter CPU2GPU;
	auto inImage = CPU2GPU.getGPUObjectsFromAssets(&inImage_CPU, &inImage_CPU + 1, cpu2gpuParams);
	assert(inImage);

	// Create an image view for input image
	core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
	{
		video::IGPUImageView::SCreationParams viewParams;
		viewParams.format = asset::EF_R8G8B8A8_UNORM; // inImage->getCreationParameters().format;
		viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
		viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.image = inImage->begin()[0];

		inImageView = device->createGPUImageView(std::move(viewParams));
	}
	assert(inImageView);

	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		const uint32_t writeDescriptorCount = 2u;

		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

		// image2D -- swapchain image
		{
			descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
			descriptorInfos[0].image.sampler = nullptr;
			descriptorInfos[0].desc = swapchainImageViews[i]; // shouldn't IGPUDescriptorSet hold a reference to the resources in its descriptors?

			writeDescriptorSets[0].dstSet = descriptorSets[i].get();
			writeDescriptorSets[0].binding = 0u;
			writeDescriptorSets[0].arrayElement = 0u;
			writeDescriptorSets[0].count = 1u;
			writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
			writeDescriptorSets[0].info = &descriptorInfos[0];
		}

		// image2D -- my input
		{
			descriptorInfos[1].image.imageLayout = asset::EIL_GENERAL;
			descriptorInfos[1].image.sampler = nullptr;
			descriptorInfos[1].desc = inImageView;

			writeDescriptorSets[1].dstSet = descriptorSets[i].get();
			writeDescriptorSets[1].binding = 1u;
			writeDescriptorSets[1].arrayElement = 0u;
			writeDescriptorSets[1].count = 1u;
			writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_IMAGE;
			writeDescriptorSets[1].info = &descriptorInfos[1];
		}

		device->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
	}

	asset::SPushConstantRange pcRange = {};
	pcRange.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
	pcRange.offset = 0u;
	pcRange.size = 2*sizeof(uint32_t);
	core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
		device->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(dsLayout));

	core::smart_refctd_ptr<video::IGPUComputePipeline> pipeline =
		device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),
			core::smart_refctd_ptr(specializedShader));


	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT];
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		acquireSemaphores[i] = device->createSemaphore();
		releaseSemaphores[i] = device->createSemaphore();
		frameFences[i] = device->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
	}

	// Record commands in commandBuffers here
	const uint32_t windowDim[2] = { window->getWidth() / 2, window->getHeight() };
	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		video::IGPUCommandBuffer::SImageMemoryBarrier undefToComputeTransitionBarrier;
		undefToComputeTransitionBarrier.barrier.srcAccessMask = asset::EAF_TRANSFER_READ_BIT;
		undefToComputeTransitionBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
		undefToComputeTransitionBarrier.oldLayout = asset::EIL_UNDEFINED;
		undefToComputeTransitionBarrier.newLayout = asset::EIL_GENERAL;
		undefToComputeTransitionBarrier.srcQueueFamilyIndex = ~0u;
		undefToComputeTransitionBarrier.dstQueueFamilyIndex = ~0u;
		undefToComputeTransitionBarrier.image = *(swapchainImages.begin() + i);
		undefToComputeTransitionBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		undefToComputeTransitionBarrier.subresourceRange.baseMipLevel = 0u;
		undefToComputeTransitionBarrier.subresourceRange.levelCount = 1u;
		undefToComputeTransitionBarrier.subresourceRange.baseArrayLayer = 0u;
		undefToComputeTransitionBarrier.subresourceRange.layerCount = 1u;

		video::IGPUCommandBuffer::SImageMemoryBarrier computeToPresentTransitionBarrier;
		computeToPresentTransitionBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		computeToPresentTransitionBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		computeToPresentTransitionBarrier.oldLayout = asset::EIL_GENERAL;
		computeToPresentTransitionBarrier.newLayout = asset::EIL_PRESENT_SRC_KHR;
		computeToPresentTransitionBarrier.srcQueueFamilyIndex = ~0u;
		computeToPresentTransitionBarrier.dstQueueFamilyIndex = ~0u;
		computeToPresentTransitionBarrier.image = *(swapchainImages.begin() + i);
		computeToPresentTransitionBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		computeToPresentTransitionBarrier.subresourceRange.baseMipLevel = 0u;
		computeToPresentTransitionBarrier.subresourceRange.levelCount = 1u;
		computeToPresentTransitionBarrier.subresourceRange.baseArrayLayer = 0u;
		computeToPresentTransitionBarrier.subresourceRange.layerCount = 1u;

		commandBuffers[i]->begin(0);

		// Todo(achal): The fact that this pipeline barrier is solely on a compute queue might
		// affect the srcStageMask. More precisely, I think, for some reason, that
		// VK_PIPELINE_STAGE_TRANSFER_BIT shouldn't be specified in compute queue
		// but present queue (or transfer queue if theres one??)
		commandBuffers[i]->pipelineBarrier(asset::EPSF_TRANSFER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT, static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr, 1u,
			&undefToComputeTransitionBarrier);

		commandBuffers[i]->bindComputePipeline(pipeline.get());

		const video::IGPUDescriptorSet* tmp[] = { descriptorSets[i].get() };
		commandBuffers[i]->bindDescriptorSets(asset::EPBP_COMPUTE, pipelineLayout.get(),
			0u, 1u, tmp);

		commandBuffers[i]->pushConstants(pipelineLayout.get(), pcRange.stageFlags, pcRange.offset, pcRange.size, windowDim);
		commandBuffers[i]->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);

		commandBuffers[i]->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr, 1u, &computeToPresentTransitionBarrier);

		commandBuffers[i]->end();
	}

	video::ISwapchain* rawPointerToSwapchain = swapchain.get();
	
	uint32_t currentFrameIndex = 0u;
	while (eventCallback->isWindowOpen())
	{
		video::IGPUSemaphore* acquireSemaphore_frame = acquireSemaphores[currentFrameIndex].get();
		video::IGPUSemaphore* releaseSemaphore_frame = releaseSemaphores[currentFrameIndex].get();
		video::IGPUFence* fence_frame = frameFences[currentFrameIndex].get();

		video::IGPUFence::E_STATUS retval = device->waitForFences(1u, &fence_frame, true, ~0ull);
		assert(retval == video::IGPUFence::ES_SUCCESS);

		uint32_t imageIndex;
		swapchain->acquireNextImage(~0ull, acquireSemaphores[currentFrameIndex].get(), nullptr,
			&imageIndex);

		// Make sure you unsignal the fence before expecting vkQueueSubmit to signal it
		// once it finishes its execution
		device->resetFences(1u, &fence_frame);

		// Todo(achal): Not really sure why are waiting at this pipeline stage for
		// acquiring the image to render
		asset::E_PIPELINE_STAGE_FLAGS waitDstStageFlags = asset::E_PIPELINE_STAGE_FLAGS::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
		
		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.waitSemaphoreCount = 1u;
		submitInfo.pWaitSemaphores = &acquireSemaphore_frame;
		submitInfo.pWaitDstStageMask = &waitDstStageFlags;
		submitInfo.signalSemaphoreCount = 1u;
		submitInfo.pSignalSemaphores = &releaseSemaphore_frame;
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &commandBuffers[imageIndex].get();

		computeQueue->submit(1u, &submitInfo, fence_frame);

		video::IGPUQueue::SPresentInfo presentInfo;
		presentInfo.waitSemaphoreCount = 1u;
		presentInfo.waitSemaphores = &releaseSemaphore_frame;
		presentInfo.swapchainCount = 1u;
		presentInfo.swapchains = &rawPointerToSwapchain;
		presentInfo.imgIndices = &imageIndex;

		presentQueue->present(presentInfo);

		currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
	}

	device->waitIdle();

	return 0;
}

#if 0
int main()
{
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
		write[1].info = info + 1;
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

}
#endif