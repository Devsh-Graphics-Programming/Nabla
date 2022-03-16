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
	
	GPU_PIPELINE_HASH_CONTAINER gpuPipelinesPly;	
	DependentDrawData plyDrawData;

	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_rasterizeDescriptorSet;
	core::smart_refctd_ptr<video::IGPUImage> m_visbuffer;
	core::smart_refctd_ptr<video::IGPUImageView> m_visbufferView;

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

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_R8G8B8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

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

			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(),video::IGPUCommandBuffer::EL_PRIMARY,1u,&cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf);
			cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
			cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].semaphore = &gpuTransferSemaphore;
			
			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(),video::IGPUCommandBuffer::EL_PRIMARY,1u,&cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf);
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

		auto rasterizerShader = getSpecializedShader("../rasterizer.comp");

		{
			const uint32_t bindingCount = 2u;
			video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {
				// Binding 0: outImage
				{0u, asset::EDT_STORAGE_IMAGE, 1u, asset::IShader::ESS_COMPUTE, nullptr},
				// Binding 1: u_pointCloud
				{1u, asset::EDT_STORAGE_BUFFER, 1u, asset::IShader::ESS_COMPUTE, nullptr}
			};

			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout =
				logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

			const uint32_t descriptorPoolSizeCount = 2u;
			video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount] = {
				{asset::EDT_STORAGE_IMAGE, 1},
				{asset::EDT_STORAGE_BUFFER, 1}
			};

			video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
				static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);
			core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
				= logicalDevice->createDescriptorPool(descriptorPoolFlags, 1,
					descriptorPoolSizeCount, poolSizes);

			m_rasterizeDescriptorSet = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(dsLayout));
		}

		// Create the point cloud visbuffer image
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
			m_visbuffer = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(imageParams));
			
			video::IGPUImageView::SCreationParams imgViewInfo;
			{
				imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
				imgViewInfo.image = m_visbuffer;
				imgViewInfo.viewType = video::IGPUImageView::ET_2D;
				imgViewInfo.format = asset::E_FORMAT::EF_R32_UINT;
				imgViewInfo.subresourceRange.aspectMask = static_cast<asset::IImage::E_ASPECT_FLAGS>(0u);
				imgViewInfo.subresourceRange.baseMipLevel = 0;
				imgViewInfo.subresourceRange.levelCount = 1;
				imgViewInfo.subresourceRange.baseArrayLayer = 0;
				imgViewInfo.subresourceRange.layerCount = 1;
			}
			m_visbufferView = logicalDevice->createGPUImageView(std::move(imgViewInfo));
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

		auto cpuBundlePLYData = loadAndGetCpuMesh(sharedInputCWD / "ply/Spanner-ply.ply");

		core::smart_refctd_ptr<asset::ICPUMesh> cpuMeshPly = cpuBundlePLYData.first;
		auto metadataPly = cpuBundlePLYData.second->selfCast<const asset::CPLYMetadata>();

#ifdef WRITE_ASSETS
		{
			asset::IAssetWriter::SAssetWriteParams wp(cpuMeshPly.get());
			bool status = assetManager->writeAsset("Spanner_ply.ply", wp);
			assert(status);
		}
#endif // WRITE_ASSETS

		auto gpuUBODescriptorPool = createDescriptorPool(1, asset::EDT_UNIFORM_BUFFER);

		/*
			For the testing puposes we can safely assume all meshbuffers within mesh loaded from PLY & STL has same DS1 layout (used for camera-specific data)
		*/

		auto getMeshDependentDrawData = [&](core::smart_refctd_ptr<asset::ICPUMesh> cpuMesh) -> DependentDrawData
		{
			const asset::ICPUMeshBuffer* const firstMeshBuffer = cpuMesh->getMeshBuffers().begin()[0];
			const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u); //! DS1
			const asset::IRenderpassIndependentPipelineMetadata* pipelineMetadata;
			pipelineMetadata = metadataPly->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

			/*
				So we can create just one DescriptorSet
			*/

			auto getDS1UboBinding = [&]()
			{
				uint32_t ds1UboBinding = 0u;
				for (const auto& bnd : ds1layout->getBindings())
					if (bnd.type == asset::EDT_UNIFORM_BUFFER)
					{
						ds1UboBinding = bnd.binding;
						break;
					}
				return ds1UboBinding;
			};

			const uint32_t ds1UboBinding = getDS1UboBinding();

			auto getNeededDS1UboByteSize = [&]()
			{
				size_t neededDS1UboSize = 0ull;
				{
					for (const auto& shaderInputs : pipelineMetadata->m_inputSemantics)
						if (shaderInputs.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shaderInputs.descriptorSection.uniformBufferObject.set == 1u && shaderInputs.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
							neededDS1UboSize = std::max<size_t>(neededDS1UboSize, shaderInputs.descriptorSection.uniformBufferObject.relByteoffset + shaderInputs.descriptorSection.uniformBufferObject.bytesize);
				}
				return neededDS1UboSize;
			};

			const uint64_t uboDS1ByteSize = getNeededDS1UboByteSize();

			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
			{
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
				if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				gpuds1layout = (*gpu_array)[0];
			}

			auto ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
			ubomemreq.vulkanReqs.size = uboDS1ByteSize;

			video::IGPUBuffer::SCreationParams creationParams;
			creationParams.canUpdateSubRange = true;
			creationParams.usage = asset::IBuffer::E_USAGE_FLAGS::EUF_UNIFORM_BUFFER_BIT;
			creationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
			creationParams.queueFamilyIndices = 0u;
			creationParams.queueFamilyIndices = nullptr;

			auto gpuubo = logicalDevice->createGPUBufferOnDedMem(creationParams, ubomemreq);
			auto gpuds1 = logicalDevice->createGPUDescriptorSet(gpuUBODescriptorPool.get(), std::move(gpuds1layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet write;
				write.dstSet = gpuds1.get();
				write.binding = ds1UboBinding;
				write.count = 1u;
				write.arrayElement = 0u;
				write.descriptorType = asset::EDT_UNIFORM_BUFFER;
				video::IGPUDescriptorSet::SDescriptorInfo info;
				{
					info.desc = gpuubo;
					info.buffer.offset = 0ull;
					info.buffer.size = uboDS1ByteSize;
				}
				write.info = &info;
				logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
			}

			core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
			{
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuMesh.get(), &cpuMesh.get() + 1, cpu2gpuParams);
				cpu2gpuParams.waitForCreationToComplete(true);
				cpu2gpuParams.beginCommandBuffers();
				if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				gpumesh = (*gpu_array)[0];
			}

			return std::make_tuple(gpumesh, gpuubo, gpuds1, ds1UboBinding, pipelineMetadata);
		};

		plyDrawData = getMeshDependentDrawData(cpuMeshPly);

		{
			auto fillGpuPipeline = [&](GPU_PIPELINE_HASH_CONTAINER& container, video::IGPUMesh* gpuMesh)
			{
				for (size_t i = 0; i < gpuMesh->getMeshBuffers().size(); ++i)
				{
					auto gpuIndependentPipeline = gpuMesh->getMeshBuffers().begin()[i]->getPipeline();

					nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
					graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
					graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

					const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
					container[adress] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
				}
			};

			fillGpuPipeline(gpuPipelinesPly, std::get<core::smart_refctd_ptr<video::IGPUMesh>>(plyDrawData).get());
		}

		// Get the positions from the mesh
		core::smart_refctd_ptr<video::IGPUOffsetBufferPair> positionsVertexBuffer;
		{
			const asset::ICPUMeshBuffer* const firstMeshBuffer = cpuMeshPly->getMeshBuffers().begin()[0];
			const asset::SBufferBinding<const asset::ICPUBuffer> posBufferBinding = firstMeshBuffer->getVertexBufferBindings()[firstMeshBuffer->getPositionAttributeIx()];
			core::smart_refctd_ptr<const asset::ICPUBuffer> posBuffer = posBufferBinding.buffer;
			auto posBuffer_cpu = posBuffer.get();
			positionsVertexBuffer = cpu2gpu.getGPUObjectsFromAssets(&posBuffer_cpu, &posBuffer_cpu + 1, cpu2gpuParams)->front();
		}

		// Fill out the descriptor sets
		{
			const uint32_t writeDescriptorCount = 2u;

			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
			video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

			// Visbuffer
			{
				descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[0].image.sampler = nullptr;
				descriptorInfos[0].desc = m_visbufferView;

				writeDescriptorSets[0].dstSet = m_rasterizeDescriptorSet.get();
				writeDescriptorSets[0].binding = 0u;
				writeDescriptorSets[0].arrayElement = 0u;
				writeDescriptorSets[0].count = 1u;
				writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[0].info = &descriptorInfos[0];
			}

			// Point cloud vertex buffer
			{
				descriptorInfos[1].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[1].image.sampler = nullptr;
				// TODO
				// descriptorInfos[1].desc = positionsVertexBuffer;

				writeDescriptorSets[1].dstSet = m_rasterizeDescriptorSet.get();
				writeDescriptorSets[1].binding = 0u;
				writeDescriptorSets[1].arrayElement = 0u;
				writeDescriptorSets[1].count = 1u;
				writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_BUFFER;
				writeDescriptorSets[1].info = &descriptorInfos[1];
			}
		}

		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.001, 1000);
		camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 0.01f, 1.f);
		lastTime = std::chrono::system_clock::now();

		for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
			dtList[i] = 0.0;

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}
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
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
		lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			time_sum -= dtList[frame_count];
			time_sum += renderDt;
			dtList[frame_count] = renderDt;
			frame_count++;
			if (frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				frameDataFilled = true;
				frame_count = 0;
			}

		}
		const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const ui::IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const ui::IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
		camera.endInputProcessing(nextPresentationTimeStamp);

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

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 1.f;
			clear[0].color.float32[2] = 1.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbos[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		auto renderMesh = [&](GPU_PIPELINE_HASH_CONTAINER& gpuPipelines, DependentDrawData& drawData, uint32_t index)
		{
			auto gpuMesh = std::get<core::smart_refctd_ptr<video::IGPUMesh>>(drawData);
			auto gpuubo = std::get<core::smart_refctd_ptr<video::IGPUBuffer>>(drawData);
			auto gpuds1 = std::get<core::smart_refctd_ptr<video::IGPUDescriptorSet>>(drawData);
			auto ds1UboBinding = std::get<uint32_t>(drawData);
			const auto* pipelineMetadata = std::get<const asset::IRenderpassIndependentPipelineMetadata*>(drawData);

			core::matrix3x4SIMD modelMatrix;

			if (index == 1)
				modelMatrix.setScale(core::vectorSIMDf(10, 10, 10));
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(index * 150, 0, 0, 0));

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

			core::vector<uint8_t> uboData(gpuubo->getSize());
			for (const auto& shaderInputs : pipelineMetadata->m_inputSemantics)
			{
				if (shaderInputs.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shaderInputs.descriptorSection.uniformBufferObject.set == 1u && shaderInputs.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
				{
					switch (shaderInputs.type)
					{
					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
					{
						memcpy(uboData.data() + shaderInputs.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shaderInputs.descriptorSection.uniformBufferObject.bytesize);
					} break;

					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
					{
						memcpy(uboData.data() + shaderInputs.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shaderInputs.descriptorSection.uniformBufferObject.bytesize);
					} break;

					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
					{
						memcpy(uboData.data() + shaderInputs.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shaderInputs.descriptorSection.uniformBufferObject.bytesize);
					} break;
					}
				}
			}

			commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

			for (auto gpuMeshBuffer : gpuMesh->getMeshBuffers())
			{
				auto gpuGraphicsPipeline = gpuPipelines[reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];

				const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
				const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();

				commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

				const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
				commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, 0u);
				const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();

				if (gpuds3_ptr)
					commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr, 0u);
				commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

				commandBuffer->drawMeshBuffer(gpuMeshBuffer);
			}
		};

		/*
			Record PLY and STL rendering commands
		*/

		renderMesh(gpuPipelinesPly, plyDrawData, 0);

		commandBuffer->endRenderPass();
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