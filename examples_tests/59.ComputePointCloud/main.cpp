// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "common.h"

using namespace nbl;
using namespace core;

/*
	Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

/*
	Uncomment for writing assets
*/

#define WRITE_ASSETS

class PointCloudRasterizer : public ApplicationBase
{
	static constexpr uint32_t WIN_W = 1280;
	static constexpr uint32_t WIN_H = 720;
	static constexpr uint32_t SC_IMG_COUNT = 3u;
	static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
	using GPU_PIPELINE_HASH_CONTAINER = std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>>;
	using DependentDrawData = std::tuple<core::smart_refctd_ptr<video::IGPUMesh>, core::smart_refctd_ptr<video::IGPUBuffer>, core::smart_refctd_ptr<video::IGPUDescriptorSet>, uint32_t, const asset::IRenderpassIndependentPipelineMetadata*>;

public:
	nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
	nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCallback;
	nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
	nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
	nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	nbl::video::IPhysicalDevice* gpuPhysicalDevice;
	std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues = { nullptr, nullptr, nullptr, nullptr };
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
	nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
	nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

	nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
	nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

	video::CDumbPresentationOracle oracle;

	uint32_t acquiredNextFBO = {};
	int resourceIx = -1;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

	std::chrono::system_clock::time_point lastTime;
	bool frameDataFilled = false;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[NBL_FRAMES_TO_AVERAGE] = {};

	CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> keyboard;

	Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_rasterizeDescriptorSet;
	// - Clear for spinlock path (Clears the visbuffer)
	// - Clear for Z prepass path (Clears the depth buffer)
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_clearDescriptorSets[2];
	// - Rasterizer for spinlock path (Rasterizes with spinlock)
	// - Depth buffer rasterizer for Z prepass path
	// - Rasterizer for Z prepass path (Uses Z prepass)
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_rasterizerPipelines[3];
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_clearPipeline;
	// - Shading for spinlock path
	// - Shading for Z prepass path
	core::smart_refctd_ptr<video::IGPUComputePipeline> m_shadingPipelines[2];

	bool useSpinlockPath = true;

	core::smart_refctd_ptr<video::IGPUImage> m_visbuffer;
	core::smart_refctd_ptr<video::IGPUImageView> m_visbufferViewRender;

	core::smart_refctd_ptr<video::IGPUImage> m_spinlock;
	core::smart_refctd_ptr<video::IGPUImageView> m_spinlockView;

	core::vector<core::smart_refctd_ptr<video::IGPUDescriptorSet>> m_shadingDescriptorSets;
	core::vector<core::smart_refctd_ptr<video::IGPUImageView>> m_swapchainImageViews;

	core::smart_refctd_ptr<video::IGPUBuffer> m_debugBuffer;

	uint32_t m_pointCount;
	uint32_t optimalWorkgroupCount;

	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return gl.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbos[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

APP_CONSTRUCTOR(PointCloudRasterizer)

	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		initOutput.system = core::smart_refctd_ptr(system);

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_UNORM, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL/*Vulkan doesn't work yet*/, "pointcloudrasterizer", WIN_W, WIN_H, SC_IMG_COUNT, swapchainImageUsage, surfaceFormat, nbl::asset::EF_D32_SFLOAT);
		window = std::move(initOutput.window);
		gl = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		system = std::move(initOutput.system);
		windowCallback = std::move(initOutput.windowCb);
		utilities = std::move(initOutput.utilities);

		auto createDescriptorPool = [&](const uint32_t count, asset::E_DESCRIPTOR_TYPE type)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
				poolSize.count = count;
				poolSize.type = type;
				return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
			}
		};

		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> gpuTransferSemaphore;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUFence> gpuComputeFence;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> gpuComputeSemaphore;

		{
			gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
			gpuTransferSemaphore = logicalDevice->createSemaphore();

			gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
			gpuComputeSemaphore = logicalDevice->createSemaphore();

			cpu2gpuParams.utilities = utilities.get();
			cpu2gpuParams.device = logicalDevice.get();
			cpu2gpuParams.assetManager = assetManager.get();
			cpu2gpuParams.pipelineCache = nullptr;
			cpu2gpuParams.limits = gpuPhysicalDevice->getLimits();
			cpu2gpuParams.finalQueueFamIx = queues[decltype(initOutput)::EQT_GRAPHICS]->getFamilyIndex();
			cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;

			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf);
			cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
			cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].semaphore = &gpuTransferSemaphore;

			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf);
			cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queues[decltype(initOutput)::EQT_COMPUTE];
			cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].semaphore = &gpuComputeSemaphore;

			cpu2gpuParams.beginCommandBuffers();
		}

		// Load the shader and its descriptor sets
		auto getSpecializedShader = [&](const char* pathToShader) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
		{
			core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
			{
				asset::IAssetLoader::SAssetLoadParams params = {};
				params.logger = logger.get();
				auto spec = (assetManager->getAsset(pathToShader, params).getContents());
				auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(pathToShader, params).getContents().begin());
				specializedShader = cpu2gpu.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, cpu2gpuParams)->front();
			}
			assert(specializedShader);

			return specializedShader;
		};

		// TODO use an automatic way of generating these variations
		auto spinlockRasterShader = getSpecializedShader("../rasterizer_spinlock.comp");
		auto zprepassRasterShader = getSpecializedShader("../rasterizer_genZprepass.comp");
		auto useZprepassRasterShader = getSpecializedShader("../rasterizer_usingZprepass.comp");

		auto shadingStdShader = getSpecializedShader("../shading_std.comp");
		auto shadingZprepassShader = getSpecializedShader("../shading_zprepass.comp");

		auto clearShader = getSpecializedShader("../clear.comp");

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> rasterizerDsLayout;
		{
			// Rasterizer descriptor set layout
			const uint32_t bindingCount = 3u;
			video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {
				// Binding 0: outImage
				{0u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr},
				// Binding 1: spinlock
				{1u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr},
				// Binding 2: u_pointCloud
				{2u, asset::EDT_STORAGE_BUFFER, 1u, asset::IShader::ESS_COMPUTE, nullptr},
			};

			rasterizerDsLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

			const uint32_t descriptorPoolSizeCount = 2u;
			video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount] = {
				{asset::EDT_STORAGE_IMAGE, 2},
				{asset::EDT_STORAGE_BUFFER, 1}
			};

			video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
				static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);
			core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
				= logicalDevice->createDescriptorPool(descriptorPoolFlags, 1,
					descriptorPoolSizeCount, poolSizes);

			m_rasterizeDescriptorSet = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(rasterizerDsLayout));
		}

		const uint32_t swapchainImageCount = swapchain->getImageCount();

		m_swapchainImageViews.resize(swapchainImageCount);
		for (uint32_t i = 0; i < swapchainImageCount; i++)
		{
			auto img = swapchain->getImages().begin()[i];
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

				m_swapchainImageViews[i] = logicalDevice->createGPUImageView(std::move(viewParams));
				assert(m_swapchainImageViews[i]);
			}
		}

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> shadingDsLayout;
		{
			// Shading descriptor set layout
			const uint32_t bindingCount = 4u;
			video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {
				// Binding 0: visbufferInput
				{0u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr},
				// Binding 1: u_pointCloud
				{1u, asset::EDT_STORAGE_BUFFER, 1u, asset::IShader::ESS_COMPUTE, nullptr},
				// Binding 2: swapchainOutput
				{2u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr},
				// Binding 0: depthBuffer
				{3u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr}
			};

			shadingDsLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

			const uint32_t descriptorPoolSizeCount = 2u;
			video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount] = {
				{asset::EDT_STORAGE_IMAGE, 3},
				{asset::EDT_STORAGE_BUFFER, 1},
			};

			video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
				static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);
			core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
				= logicalDevice->createDescriptorPool(descriptorPoolFlags, swapchainImageCount,
					descriptorPoolSizeCount, poolSizes);

			m_shadingDescriptorSets.resize(swapchainImageCount);
			for (uint32_t i = 0; i < swapchainImageCount; i++)
			{
				m_shadingDescriptorSets[i] = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(shadingDsLayout));
			}
		}

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> clearDsLayout;
		{
			// Clear descriptor set layout
			const uint32_t bindingCount = 1u;
			video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {
				// Binding 0: visbuffer
				{0u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr},
			};

			clearDsLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

			const uint32_t descriptorPoolSizeCount = 2u;
			video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount] = {
				{asset::EDT_STORAGE_IMAGE, 1},
			};

			video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
				static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);
			core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
				= logicalDevice->createDescriptorPool(descriptorPoolFlags, swapchainImageCount,
					descriptorPoolSizeCount, poolSizes);

			m_clearDescriptorSets[0] = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(clearDsLayout));
			m_clearDescriptorSets[1] = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(clearDsLayout));
		}

		auto getImgView = [&](core::smart_refctd_ptr<video::IGPUImage> image, asset::E_FORMAT format) -> core::smart_refctd_ptr<video::IGPUImageView>
		{
			video::IGPUImageView::SCreationParams imgViewInfo;
			imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
			imgViewInfo.image = image;
			imgViewInfo.viewType = video::IGPUImageView::ET_2D;
			imgViewInfo.subresourceRange.aspectMask = static_cast<asset::IImage::E_ASPECT_FLAGS>(asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT);
			imgViewInfo.subresourceRange.baseMipLevel = 0;
			imgViewInfo.subresourceRange.levelCount = 1;
			imgViewInfo.subresourceRange.baseArrayLayer = 0;
			imgViewInfo.subresourceRange.layerCount = 1;
			imgViewInfo.format = format;
			return logicalDevice->createGPUImageView(std::move(imgViewInfo));
		};

		// Create the point cloud visbuffer image
		{
			video::IGPUImage::SCreationParams imageParams;
			{
				imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
				imageParams.format = asset::E_FORMAT::EF_R32G32_UINT;
				imageParams.type = asset::IImage::E_TYPE::ET_2D;
				imageParams.samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
				imageParams.extent = { WIN_W, WIN_H, 1 };
				imageParams.mipLevels = 1;
				imageParams.arrayLayers = 1;
			}
			m_visbuffer = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(imageParams));
			m_visbufferViewRender = getImgView(m_visbuffer, asset::E_FORMAT::EF_R32G32_UINT);
		}

		// Create the spinlock image
		{
			video::IGPUImage::SCreationParams imageParams;
			{
				imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
				imageParams.format = asset::E_FORMAT::EF_R32_UINT;
				imageParams.type = asset::IImage::E_TYPE::ET_2D;
				imageParams.samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
				imageParams.extent = { WIN_W, WIN_H, 1 };
				imageParams.mipLevels = 1;
				imageParams.arrayLayers = 1;
			}
			m_spinlock = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(imageParams));
			m_spinlockView = getImgView(m_spinlock, asset::E_FORMAT::EF_R32_UINT);
		}

		// Load the mesh
		auto loadAndGetCpuMesh = [&](system::path path) -> std::pair<core::smart_refctd_ptr<asset::ICPUMesh>, const asset::IAssetMetadata*>
		{
			auto meshes_bundle = assetManager->getAsset(path.string(), {});
			{
				bool status = !meshes_bundle.getContents().empty();
				assert(status);
			}

			auto mesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshes_bundle.getContents().begin()[0]);
			auto metadata = meshes_bundle.getMetadata();
			return std::make_pair(mesh, metadata);
			//return std::make_pair(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshes_bundle.getContents().begin()[0]), meshes_bundle.getMetadata());
		};

		auto cpuBundlePLYData = loadAndGetCpuMesh(sharedInputCWD / "ply/Ruddington_Neighbrhd_2022-03-29_10h19_10_611.ply");

		core::smart_refctd_ptr<asset::ICPUMesh> cpuMeshPly = cpuBundlePLYData.first;
		auto metadataPly = cpuBundlePLYData.second->selfCast<const asset::CPLYMetadata>();

		// Get the positions from the mesh
		uint32_t positionVbOffset;
		core::smart_refctd_ptr<video::IGPUBuffer> positionVb;

		{
			auto meshManipulator = assetManager->getMeshManipulator();

			const asset::ICPUMeshBuffer* const firstMeshBuffer = cpuMeshPly->getMeshBuffers().begin()[0];
			core::smart_refctd_ptr<asset::ICPUMeshBuffer> meshBuffer = core::smart_refctd_ptr_static_cast<asset::ICPUMeshBuffer>(firstMeshBuffer->clone());

			meshBuffer->setIndexBufferBinding({});
			for (uint32_t binding = 0; binding < asset::ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT; binding++)
			{
				if (binding != meshBuffer->getPositionAttributeIx()) 
				{
					meshBuffer->setVertexBufferBinding({}, binding);
				}
			}

			const asset::SBufferBinding<asset::ICPUBuffer> posBufferBinding = meshBuffer->getVertexBufferBindings()[meshBuffer->getPositionAttributeIx()];
			core::smart_refctd_ptr<const asset::ICPUBuffer> posBuffer = posBufferBinding.buffer;

			m_pointCount = posBuffer->getSize() / 12;
			optimalWorkgroupCount = logicalDevice->getPhysicalDevice()->getLimits().computeOptimalPersistentWorkgroupDispatchSize(m_pointCount, 256);

			auto posBuffer_cpu = posBuffer.get();
			auto positionsVertexBuffer = cpu2gpu.getGPUObjectsFromAssets(&posBuffer_cpu, &posBuffer_cpu + 1, cpu2gpuParams)->front();
			positionVbOffset = positionsVertexBuffer->getOffset();
			positionVb = core::smart_refctd_ptr<video::IGPUBuffer>(positionsVertexBuffer->getBuffer());
		}

		// Fill out the descriptor sets
		// Rasterizer descriptor sets
		{
			const uint32_t writeDescriptorCount = 3u;

			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
			video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

			// Visbuffer
			{
				descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[0].image.sampler = nullptr;
				descriptorInfos[0].desc = m_visbufferViewRender;

				writeDescriptorSets[0].dstSet = m_rasterizeDescriptorSet.get();
				writeDescriptorSets[0].binding = 0u;
				writeDescriptorSets[0].arrayElement = 0u;
				writeDescriptorSets[0].count = 1u;
				writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[0].info = &descriptorInfos[0];
			}

			// Spinlock image
			{
				descriptorInfos[1].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[1].image.sampler = nullptr;
				descriptorInfos[1].desc = m_spinlockView;

				writeDescriptorSets[1].dstSet = m_rasterizeDescriptorSet.get();
				writeDescriptorSets[1].binding = 1u;
				writeDescriptorSets[1].arrayElement = 0u;
				writeDescriptorSets[1].count = 1u;
				writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[1].info = &descriptorInfos[1];
			}

			// Point cloud vertex buffer
			{
				descriptorInfos[2].buffer.offset = positionVbOffset;
				descriptorInfos[2].buffer.size = positionVb->getSize() - positionVbOffset;
				descriptorInfos[2].desc = positionVb;

				writeDescriptorSets[2].dstSet = m_rasterizeDescriptorSet.get();
				writeDescriptorSets[2].binding = 2u;
				writeDescriptorSets[2].arrayElement = 0u;
				writeDescriptorSets[2].count = 1u;
				writeDescriptorSets[2].descriptorType = asset::EDT_STORAGE_BUFFER;
				writeDescriptorSets[2].info = &descriptorInfos[2];
			}

			logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
		}

		{
			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(RasterizerPushConstants);

			// Rasterizer pipeline
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
				logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(rasterizerDsLayout));
			m_rasterizerPipelines[0] = logicalDevice->createGPUComputePipeline(nullptr,
				core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(spinlockRasterShader));
			m_rasterizerPipelines[1] = logicalDevice->createGPUComputePipeline(nullptr,
				core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(zprepassRasterShader));
			m_rasterizerPipelines[2] = logicalDevice->createGPUComputePipeline(nullptr,
				core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(useZprepassRasterShader));
		}

		// Shading descriptor sets
		for (uint32_t i = 0u; i < swapchainImageCount; i++)
		{
			const uint32_t writeDescriptorCount = 4u;

			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
			video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

			// Visbuffer
			{
				descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[0].image.sampler = nullptr;
				descriptorInfos[0].desc = m_visbufferViewRender;

				writeDescriptorSets[0].dstSet = m_shadingDescriptorSets[i].get();
				writeDescriptorSets[0].binding = 0u;
				writeDescriptorSets[0].arrayElement = 0u;
				writeDescriptorSets[0].count = 1u;
				writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[0].info = &descriptorInfos[0];
			}

			// Point cloud vertex buffer
			{
				descriptorInfos[1].buffer.offset = positionVbOffset;
				descriptorInfos[1].buffer.size = positionVb->getSize() - positionVbOffset;
				descriptorInfos[1].desc = positionVb;

				writeDescriptorSets[1].dstSet = m_shadingDescriptorSets[i].get();
				writeDescriptorSets[1].binding = 1u;
				writeDescriptorSets[1].arrayElement = 0u;
				writeDescriptorSets[1].count = 1u;
				writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_BUFFER;
				writeDescriptorSets[1].info = &descriptorInfos[1];
			}

			// Swapchain output
			{
				descriptorInfos[2].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[2].image.sampler = nullptr;
				descriptorInfos[2].desc = m_swapchainImageViews[i];

				writeDescriptorSets[2].dstSet = m_shadingDescriptorSets[i].get();
				writeDescriptorSets[2].binding = 2u;
				writeDescriptorSets[2].arrayElement = 0u;
				writeDescriptorSets[2].count = 1u;
				writeDescriptorSets[2].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[2].info = &descriptorInfos[2];
			}

			// Depth buffer (spinlock) output
			{
				descriptorInfos[3].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[3].image.sampler = nullptr;
				descriptorInfos[3].desc = m_spinlockView;

				writeDescriptorSets[3].dstSet = m_shadingDescriptorSets[i].get();
				writeDescriptorSets[3].binding = 3u;
				writeDescriptorSets[3].arrayElement = 0u;
				writeDescriptorSets[3].count = 1u;
				writeDescriptorSets[3].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[3].info = &descriptorInfos[3];
			}

			logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
		}

		{
			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(ShadingPushConstants);

			// Shading pipeline
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
				logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(shadingDsLayout));
			m_shadingPipelines[0] = logicalDevice->createGPUComputePipeline(nullptr,
				core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(shadingStdShader));
			m_shadingPipelines[1] = logicalDevice->createGPUComputePipeline(nullptr,
				core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(shadingZprepassShader));
		}

		// Clear descriptor set
		for (uint32_t i = 0u; i < 2; i++)
		{
			const uint32_t writeDescriptorCount = 1u;

			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
			video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

			// Visbuffer
			{
				descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[0].image.sampler = nullptr;
				descriptorInfos[0].desc = i == 0 ? m_visbufferViewRender : m_spinlockView;

				writeDescriptorSets[0].dstSet = m_clearDescriptorSets[i].get();
				writeDescriptorSets[0].binding = 0u;
				writeDescriptorSets[0].arrayElement = 0u;
				writeDescriptorSets[0].count = 1u;
				writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[0].info = &descriptorInfos[0];
			}

			logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
		}

		{
			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(ShadingPushConstants);

			// Clear pipeline
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
				logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(clearDsLayout));
			m_clearPipeline = logicalDevice->createGPUComputePipeline(nullptr,
				core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(clearShader));
		}

		// Other initialization
		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.001, 1000);
		camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 0.1f, 1.f);
		lastTime = std::chrono::system_clock::now();

		for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
			dtList[i] = 0.0;

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}

		oracle.reportBeginFrameRecord();
	}

	void onAppTerminated_impl() override
	{
		const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		//TODO: 
		bool status = ext::ScreenShot::createScreenShot(
			logicalDevice.get(),
			queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
			renderFinished[resourceIx].get(),
			gpuSourceImageView.get(),
			assetManager.get(),
			"ScreenShot.png",
			asset::EIL_PRESENT_SRC,
			static_cast<asset::E_ACCESS_FLAGS>(0u));
		assert(status);
	}

	void workLoopBody() override
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];

		if (fence)
			logicalDevice->blockForFences(1u, &fence.get());
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimestamp);
		mouse.consumeEvents([&](const ui::IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		bool clearSpinlock = false;
		keyboard.consumeEvents([&](const ui::IKeyboardEventChannel::range_t& events) -> void 
		{ 
			camera.keyboardProcess(events); 

			for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
			{
				auto ev = *eventIt;
				// "G" for toggling modes
				if (ev.keyCode == nbl::ui::EKC_G) {
					useSpinlockPath = !useSpinlockPath;
					clearSpinlock = true;
				}
			}
		}, logger.get());
		camera.endInputProcessing(nextPresentationTimestamp);

		const auto& viewMatrix = camera.getViewMatrix();
		const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);

		swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		core::matrix3x4SIMD modelMatrix;
		core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

		// For transitioning the swapchain
		video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier;
		{
			layoutTransBarrier.srcQueueFamilyIndex = ~0u;
			layoutTransBarrier.dstQueueFamilyIndex = ~0u;
			layoutTransBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
			layoutTransBarrier.subresourceRange.levelCount = 1u;
			layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
			layoutTransBarrier.subresourceRange.layerCount = 1u;
			layoutTransBarrier.image = *(swapchain->getImages().begin() + acquiredNextFBO);
		}

		// Clear the visbuffer
		auto runClearPath = [&](uint32_t setIdx)
		{
			commandBuffer->bindComputePipeline(m_clearPipeline.get());

			commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, m_clearPipeline->getLayout(), 0u, 1u, &m_clearDescriptorSets[setIdx].get());

			const asset::SPushConstantRange& pcRange = m_clearPipeline->getLayout()->getPushConstantRanges().begin()[0];
			ShadingPushConstants pushConstants;
			pushConstants.imgSize = { window->getWidth(), window->getHeight() };

			commandBuffer->pushConstants(m_clearPipeline->getLayout(), pcRange.stageFlags, pcRange.offset, pcRange.size, &pushConstants);
			commandBuffer->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);
		};

		auto runRasterPath = [&](uint32_t pipelineIdx)
		{
			commandBuffer->bindComputePipeline(m_rasterizerPipelines[pipelineIdx].get());

			commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, m_rasterizerPipelines[pipelineIdx]->getLayout(), 0u, 1u, &m_rasterizeDescriptorSet.get());

			const asset::SPushConstantRange& pcRange = m_rasterizerPipelines[pipelineIdx]->getLayout()->getPushConstantRanges().begin()[0];
			RasterizerPushConstants pushConstants;
			pushConstants.imgSize = { window->getWidth(), window->getHeight() };
			pushConstants.pointCount = m_pointCount;
			pushConstants.totalThreads = optimalWorkgroupCount * 256;
			memcpy(&pushConstants.mvp, mvp.pointer(), sizeof(core::matrix4SIMD));

			commandBuffer->pushConstants(m_rasterizerPipelines[pipelineIdx]->getLayout(), pcRange.stageFlags, pcRange.offset, pcRange.size, &pushConstants);
			commandBuffer->dispatch(optimalWorkgroupCount, 1u, 1u);
		};

		auto runShadingPass = [&](uint32_t pipelineIdx)
		{
			commandBuffer->bindComputePipeline(m_shadingPipelines[pipelineIdx].get());

			commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, m_shadingPipelines[pipelineIdx]->getLayout(), 0u, 1u, &m_shadingDescriptorSets[acquiredNextFBO].get());

			const asset::SPushConstantRange& pcRange = m_shadingPipelines[pipelineIdx]->getLayout()->getPushConstantRanges().begin()[0];
			ShadingPushConstants pushConstants;
			pushConstants.imgSize = { window->getWidth(), window->getHeight() };

			commandBuffer->pushConstants(m_shadingPipelines[pipelineIdx]->getLayout(), pcRange.stageFlags, pcRange.offset, pcRange.size, &pushConstants);
			commandBuffer->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);
		};

		// Transition swapchain for shading
		{
			layoutTransBarrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
			layoutTransBarrier.barrier.dstAccessMask = asset::EAF_SHADER_WRITE_BIT;
			layoutTransBarrier.oldLayout = asset::EIL_UNDEFINED;
			layoutTransBarrier.newLayout = asset::EIL_GENERAL;
			commandBuffer->pipelineBarrier(
				asset::EPSF_TOP_OF_PIPE_BIT,
				asset::EPSF_COMPUTE_SHADER_BIT,
				static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
				0u, nullptr,
				0u, nullptr,
				1u, &layoutTransBarrier);
		}

		if (clearSpinlock)
		{
			// Clear the spinlock buffer
			runClearPath(1);
			clearSpinlock = false;
		}

		if (useSpinlockPath) 
		{
			// Clear the visbuffer
			runClearPath(0);
			// Rasterize the visbuffer (spinlock)
			runRasterPath(0);
			// Shading
			runShadingPass(0);
		} else {
			// Clear the Z buffer
			runClearPath(1);
			// Rasterize the Z buffer
			runRasterPath(1);
			// Rasterize the visbuffer (with the Z prepass)
			runRasterPath(2);
			// Shading
			runShadingPass(1);
		}

		// Transition swapchain for present
		{
			layoutTransBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			layoutTransBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
			layoutTransBarrier.oldLayout = asset::EIL_GENERAL;
			layoutTransBarrier.newLayout = asset::EIL_PRESENT_SRC;
			commandBuffer->pipelineBarrier(
				asset::EPSF_COMPUTE_SHADER_BIT,
				asset::EPSF_BOTTOM_OF_PIPE_BIT,
				static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
				0u, nullptr,
				0u, nullptr,
				1u, &layoutTransBarrier);
		}

		commandBuffer->end();

		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCallback->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(PointCloudRasterizer)
