#define _NBL_STATIC_LIB
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;

class ClusteredRenderingSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

	int32_t resourceIx = -1;
	uint32_t acquiredNextFBO = {};

	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;

	const nbl::asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata;

	uint32_t ds1UboBinding = 0;
	core::smart_refctd_ptr<video::IGPUBuffer> gpuubo;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> gpuds1;

	core::smart_refctd_ptr<video::IGPUMesh> gpumesh;

	video::CDumbPresentationOracle oracle;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

public:
	auto createDescriptorPool(const uint32_t textureCount)
	{
		constexpr uint32_t maxItemCount = 256u;
		{
			nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
			poolSize.count = textureCount;
			poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
			return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
		}
	}
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}

	APP_CONSTRUCTOR(ClusteredRenderingSampleApp);

	void onAppInitialized_impl() override
	{
		CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};
		requiredInstanceFeatures.count = 1u;
		video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { video::IAPIConnection::EF_SURFACE };
		requiredInstanceFeatures.features = requiredFeatures_Instance;

		CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};

		CommonAPI::SFeatureRequest<video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};
		requiredDeviceFeatures.count = 1u;
		video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] = { video::ILogicalDevice::EF_SWAPCHAIN };
		requiredDeviceFeatures.features = requiredFeatures_Device;

		CommonAPI::SFeatureRequest< video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_UNORM, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::Init(
			initOutput,
			video::EAT_VULKAN,
			"ClusteredRendering",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W, WIN_H, SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat,
			asset::EF_D32_SFLOAT);

		system = std::move(initOutput.system);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
		quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

		system::path archPath = sharedInputCWD / "sponza.zip";
		auto arch = system->openFileArchive(archPath);
		// test no alias loading (TODO: fix loading from absolute paths)
		system->mount(std::move(arch));
		asset::IAssetLoader::SAssetLoadParams loadParams;
		loadParams.workingDirectory = sharedInputCWD;
		loadParams.logger = logger.get();
		auto meshes_bundle = assetManager->getAsset((sharedInputCWD / "sponza.zip/sponza.obj").string(), loadParams);
		assert(!meshes_bundle.getContents().empty());

		quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

		auto cpuMesh = meshes_bundle.getContents().begin()[0];
		asset::ICPUMesh* meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

		const asset::ICPUMeshBuffer* firstMeshBuffer;
		firstMeshBuffer = *meshRaw->getMeshBuffers().begin();
		const asset::COBJMetadata* metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();
		pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

		const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
		ds1UboBinding = 0u;
		for (const auto& bnd : ds1layout->getBindings())
		{
			if (bnd.type == asset::EDT_UNIFORM_BUFFER)
			{
				ds1UboBinding = bnd.binding;
				break;
			}
		}

		size_t neededDS1UBOsz = 0ull;
		{
			for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
				if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
					neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);
		}

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
		{
			cpu2gpuParams.beginCommandBuffers();
			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
			if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
				assert(false);
			cpu2gpuParams.waitForCreationToComplete();
			gpuds1layout = (*gpu_array)[0];
		}

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool = createDescriptorPool(1u);

		video::IDriverMemoryBacked::SDriverMemoryRequirements ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
		ubomemreq.vulkanReqs.size = neededDS1UBOsz;
		video::IGPUBuffer::SCreationParams gpuuboCreationParams;
		gpuuboCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
		gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
		gpuuboCreationParams.queueFamilyIndexCount = 0u;
		gpuuboCreationParams.queueFamilyIndices = nullptr;

		gpuubo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams, ubomemreq);
		gpuds1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuds1layout));

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
				info.buffer.size = neededDS1UBOsz;
			}
			write.info = &info;
			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
		}
		
		{
			for (size_t i = 0ull; i < meshRaw->getMeshBuffers().size(); ++i)
			{
				auto& meshBuffer = meshRaw->getMeshBuffers().begin()[i];

				// Todo(achal): Can get rid of this probably after
				// https://github.com/Devsh-Graphics-Programming/Nabla/pull/160#discussion_r747185441
				for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
					meshBuffer->getPipeline()->getBlendParams().blendParams[i].attachmentEnabled = (i == 0ull);

				meshBuffer->getPipeline()->getRasterizationParams().frontFaceIsCCW = false;
			}

			cpu2gpuParams.beginCommandBuffers();
			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
			if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
				assert(false);
			cpu2gpuParams.waitForCreationToComplete();

			gpumesh = (*gpu_array)[0];
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &bakedCommandBuffer);
		video::IGPUCommandBuffer::SInheritanceInfo inheritanceInfo = {};
		inheritanceInfo.renderpass = renderpass;
		inheritanceInfo.subpass = 0; // this should probably be kSubpassIx?
		bakedCommandBuffer->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT | video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT, &inheritanceInfo);
		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		bakedCommandBuffer->setViewport(0u, 1u, &viewport);

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W, WIN_H };
		bakedCommandBuffer->setScissor(0u, 1u, &scissor);
		{
			const uint32_t drawCallCount = gpumesh->getMeshBuffers().size();
			core::smart_refctd_ptr<video::CDrawIndirectAllocator<>> drawAllocator;
			{
				video::IDrawIndirectAllocator::ImplicitBufferCreationParameters params;
				params.device = logicalDevice.get();
				params.maxDrawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
				params.drawCommandCapacity = drawCallCount;
				params.drawCountCapacity = 0u;
				drawAllocator = video::CDrawIndirectAllocator<>::create(std::move(params));
			}
			video::IDrawIndirectAllocator::Allocation allocation;
			{
				allocation.count = drawCallCount;
				{
					allocation.multiDrawCommandRangeByteOffsets = new uint32_t[allocation.count];
					// you absolutely must do this
					std::fill_n(allocation.multiDrawCommandRangeByteOffsets, allocation.count, video::IDrawIndirectAllocator::invalid_draw_range_begin);
				}
				{
					auto drawCounts = new uint32_t[allocation.count];
					std::fill_n(drawCounts, allocation.count, 1u);
					allocation.multiDrawCommandMaxCounts = drawCounts;
				}
				allocation.setAllCommandStructSizesConstant(sizeof(asset::DrawElementsIndirectCommand_t));
				drawAllocator->allocateMultiDraws(allocation);
				delete[] allocation.multiDrawCommandMaxCounts;
			}

			video::CSubpassKiln kiln;
			constexpr auto kSubpassIx = 0u;

			auto drawCallData = new asset::DrawElementsIndirectCommand_t[drawCallCount];
			{
				auto drawIndexIt = allocation.multiDrawCommandRangeByteOffsets;
				auto drawCallDataIt = drawCallData;
				core::map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines;
				for (auto& mb : gpumesh->getMeshBuffers())
				{
					auto& drawcall = kiln.getDrawcallMetadataVector().emplace_back();
					memcpy(drawcall.pushConstantData, mb->getPushConstantsDataPtr(), video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
					{
						auto renderpassIndep = mb->getPipeline();
						auto foundPpln = graphicsPipelines.find(renderpassIndep);
						if (foundPpln == graphicsPipelines.end())
						{
							video::IGPUGraphicsPipeline::SCreationParams params;
							params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
							params.renderpass = core::smart_refctd_ptr(renderpass);
							params.subpassIx = kSubpassIx;
							foundPpln = graphicsPipelines.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
						}
						drawcall.pipeline = foundPpln->second;
					}
					drawcall.descriptorSets[1] = gpuds1;
					drawcall.descriptorSets[3] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(mb->getAttachedDescriptorSet());
					std::copy_n(mb->getVertexBufferBindings(), video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT, drawcall.vertexBufferBindings);
					drawcall.indexBufferBinding = mb->getIndexBufferBinding().buffer;
					drawcall.drawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
					drawcall.indexType = mb->getIndexType();
					//drawcall.drawCountOffset // leave as invalid
					drawcall.drawCallOffset = *(drawIndexIt++);
					drawcall.drawMaxCount = 1u;

					// TODO: in the far future, just make IMeshBuffer hold a union of `DrawArraysIndirectCommand_t` `DrawElementsIndirectCommand_t`
					drawCallDataIt->count = mb->getIndexCount();
					drawCallDataIt->instanceCount = mb->getInstanceCount();
					switch (drawcall.indexType)
					{
					case asset::EIT_32BIT:
						drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset / sizeof(uint32_t);
						break;
					case asset::EIT_16BIT:
						drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset / sizeof(uint16_t);
						break;
					default:
						assert(false);
						break;
					}
					drawCallDataIt->baseVertex = mb->getBaseVertex();
					drawCallDataIt->baseInstance = mb->getBaseInstance();
					drawCallDataIt++;
#ifdef REFERENCE
					bakedCommandBuffer->bindGraphicsPipeline(drawcall.pipeline.get());
					bakedCommandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, drawcall.pipeline->getRenderpassIndependentPipeline()->getLayout(), 1u, 3u, &drawcall.descriptorSets->get() + 1u, nullptr);
					bakedCommandBuffer->pushConstants(drawcall.pipeline->getRenderpassIndependentPipeline()->getLayout(), video::IGPUShader::ESS_FRAGMENT, 0u, video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, drawcall.pushConstantData);
					bakedCommandBuffer->drawMeshBuffer(mb);
#endif
				}
			}
			// do the transfer of drawcall structs
			{
				video::CPropertyPoolHandler::UpStreamingRequest request;
				request.destination = drawAllocator->getDrawCommandMemoryBlock();
				request.fill = false;
				request.elementSize = sizeof(asset::DrawElementsIndirectCommand_t);
				request.elementCount = drawCallCount;
				request.source.device2device = false;
				request.source.data = drawCallData;
				request.srcAddresses = nullptr; // iota 0,1,2,3,4,etc.
				request.dstAddresses = allocation.multiDrawCommandRangeByteOffsets;
				std::for_each_n(allocation.multiDrawCommandRangeByteOffsets, request.elementCount, [&](auto& handle) {handle /= request.elementSize; });

				auto upQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
				core::smart_refctd_ptr<video::IGPUCommandBuffer> tferCmdBuf;
				logicalDevice->createCommandBuffers(commandPools[decltype(initOutput)::EQT_TRANSFER_UP].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &tferCmdBuf);
				tferCmdBuf->begin(0u); // TODO some one time submit bit or something
				{
					auto* ppHandler = utilities->getDefaultPropertyPoolHandler();
					// if we did multiple transfers, we'd reuse the scratch
					asset::SBufferBinding<video::IGPUBuffer> scratch;
					{
						video::IGPUBuffer::SCreationParams scratchParams = {};
						scratchParams.canUpdateSubRange = true;
						scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
						scratch = { 0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,ppHandler->getMaxScratchSize()) };
						scratch.buffer->setObjectDebugName("Scratch Buffer");
					}
					auto* pRequest = &request;
					uint32_t waitSemaphoreCount = 0u;
					video::IGPUSemaphore* const* waitSemaphores = nullptr;
					const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
					ppHandler->transferProperties(
						utilities->getDefaultUpStreamingBuffer(), tferCmdBuf.get(), fence.get(), upQueue,
						scratch, pRequest, 1u, waitSemaphoreCount, waitSemaphores, waitStages,
						logger.get(), std::chrono::high_resolution_clock::time_point::max() // wait forever if necessary, need initialization to finish
					);
				}
				tferCmdBuf->end();
				{
					video::IGPUQueue::SSubmitInfo submit = {}; // intializes all semaphore stuff to 0 and nullptr
					submit.commandBufferCount = 1u;
					submit.commandBuffers = &tferCmdBuf.get();
					upQueue->submit(1u, &submit, fence.get());
				}
				logicalDevice->blockForFences(1u, &fence.get());
			}
			delete[] drawCallData;
			// free the draw command index list
			delete[] allocation.multiDrawCommandRangeByteOffsets;

#ifndef REFERENCE
			kiln.bake(bakedCommandBuffer.get(), renderpass.get(), kSubpassIx, drawAllocator->getDrawCommandMemoryBlock().buffer.get(), nullptr);
#endif
		}
		bakedCommandBuffer->end();

		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, 4000.f);
		camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);

		oracle.reportBeginFrameRecord();

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();

		const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		bool status = ext::ScreenShot::createScreenShot(
			logicalDevice.get(),
			queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN],
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
		{
			logicalDevice->blockForFences(1u, &fence.get());
			logicalDevice->resetFences(1u, &fence.get());
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		//
		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		// late latch input
		const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		// input
		{
			inputSystem->getDefaultMouse(&mouse);
			inputSystem->getDefaultKeyboard(&keyboard);

			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
			camera.endInputProcessing(nextPresentationTimestamp);
		}

		// update camera
		{
			const auto& viewMatrix = camera.getViewMatrix();
			const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

			core::vector<uint8_t> uboData(gpuubo->getSize());
			for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
			{
				if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
				{
					switch (shdrIn.type)
					{
					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
					{
						memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewProjectionMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
					} break;

					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
					{
						memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
					} break;

					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
					{
						memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
					} break;
					}
				}
			}
			commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());
		}

		// renderpass
		{
			asset::SViewport viewport;
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = WIN_W;
			viewport.height = WIN_H;
			commandBuffer->setViewport(0u, 1u, &viewport);

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

			commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_SECONDARY_COMMAND_BUFFERS);
			commandBuffer->executeCommands(1u, &bakedCommandBuffer.get());
			commandBuffer->endRenderPass();

			commandBuffer->end();
		}

		CommonAPI::Submit(
			logicalDevice.get(),
			swapchain.get(),
			commandBuffer.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			imageAcquire[resourceIx].get(),
			renderFinished[resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			renderFinished[resourceIx].get(),
			acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(ClusteredRenderingSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }