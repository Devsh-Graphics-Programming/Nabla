#define _NBL_STATIC_LIB
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;

class ClusteredRenderingSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static uint32_t CAMERA_DS_NUMBER = 1u;

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
			surfaceFormat);

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
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		const asset::E_FORMAT depthFormat = asset::EF_D32_SFLOAT;
		core::smart_refctd_ptr<video::IGPUImage> depthImages[CommonAPI::InitOutput::MaxSwapChainImageCount] = { nullptr };
		for (uint32_t i = 0u; i < swapchain->getImageCount(); ++i)
		{
			video::IGPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			imgParams.type = asset::IImage::ET_2D;
			imgParams.format = depthFormat;
			imgParams.extent = { WIN_W, WIN_H, 1 };
			imgParams.usage = asset::IImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT;
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = asset::IImage::ESCF_1_BIT;
			depthImages[i] = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(imgParams));
		}

		constexpr uint32_t zPrepassIndex = 0u;
		constexpr uint32_t lightingPassIndex = 1u;
		{
			constexpr uint32_t attachmentCount = 2u;

			video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[attachmentCount];
			// swapchain image color attachment
			attachments[0].initialLayout = asset::EIL_UNDEFINED;
			attachments[0].finalLayout = asset::EIL_PRESENT_SRC;
			attachments[0].format = swapchain->getCreationParameters().surfaceFormat.format;
			attachments[0].samples = asset::IImage::ESCF_1_BIT;
			attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
			attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;
			// depth attachment to be written to in the first subpass and read from (as a depthStencilAttachment) in the next
			attachments[1].initialLayout = asset::EIL_UNDEFINED;
			attachments[1].finalLayout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			attachments[1].format = asset::EF_D32_SFLOAT;
			attachments[1].samples = asset::IImage::ESCF_1_BIT;
			attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
			attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_DONT_CARE; // after the last usage of this attachment we can throw away its contents

			video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef swapchainColorAttRef;
			swapchainColorAttRef.attachment = 0u;
			swapchainColorAttRef.layout = asset::EIL_COLOR_ATTACHMENT_OPTIMAL;

			video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
			depthStencilAttRef.attachment = 1u;
			depthStencilAttRef.layout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			constexpr uint32_t subpassCount = 2u;
			video::IGPURenderpass::SCreationParams::SSubpassDescription subpasses[subpassCount] = {};

			// The Z Pre pass subpass
			subpasses[zPrepassIndex].pipelineBindPoint = asset::EPBP_GRAPHICS;
			subpasses[zPrepassIndex].depthStencilAttachment = &depthStencilAttRef;

			// The lighting subpass
			subpasses[lightingPassIndex].pipelineBindPoint = asset::EPBP_GRAPHICS;
			subpasses[lightingPassIndex].depthStencilAttachment = &depthStencilAttRef;
			subpasses[lightingPassIndex].colorAttachmentCount = 1u;
			subpasses[lightingPassIndex].colorAttachments = &swapchainColorAttRef;

			video::IGPURenderpass::SCreationParams::SSubpassDependency subpassDeps[3];

			subpassDeps[0].srcSubpass = video::IGPURenderpass::SCreationParams::SSubpassDependency::SUBPASS_EXTERNAL;
			subpassDeps[0].dstSubpass = 0u;
			subpassDeps[0].srcStageMask = asset::EPSF_BOTTOM_OF_PIPE_BIT;
			subpassDeps[0].srcAccessMask = asset::EAF_MEMORY_READ_BIT;
			subpassDeps[0].dstStageMask = asset::EPSF_LATE_FRAGMENT_TESTS_BIT;
			subpassDeps[0].dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_DEPTH_STENCIL_ATTACHMENT_READ_BIT | asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
			subpassDeps[0].dependencyFlags = asset::EDF_BY_REGION_BIT; // Todo(achal): Not sure

			subpassDeps[1].srcSubpass = 0u;
			subpassDeps[1].dstSubpass = 1u;
			subpassDeps[1].srcStageMask = asset::EPSF_LATE_FRAGMENT_TESTS_BIT;
			subpassDeps[1].srcAccessMask = asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			subpassDeps[1].dstStageMask = asset::EPSF_EARLY_FRAGMENT_TESTS_BIT;
			subpassDeps[1].dstAccessMask = asset::EAF_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
			subpassDeps[1].dependencyFlags = asset::EDF_BY_REGION_BIT; // Todo(achal): Not sure

			// Todo(achal): Not 100% sure why would I need this
			subpassDeps[2].srcSubpass = 0u;
			subpassDeps[2].dstSubpass = video::IGPURenderpass::SCreationParams::SSubpassDependency::SUBPASS_EXTERNAL;
			subpassDeps[2].srcStageMask = asset::EPSF_LATE_FRAGMENT_TESTS_BIT;
			subpassDeps[2].srcAccessMask = asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			subpassDeps[2].dstStageMask = asset::EPSF_BOTTOM_OF_PIPE_BIT;
			subpassDeps[2].dstAccessMask = asset::EAF_MEMORY_READ_BIT;
			subpassDeps[2].dependencyFlags = asset::EDF_BY_REGION_BIT; // Todo(achal): Not sure

			video::IGPURenderpass::SCreationParams creationParams = {};
			creationParams.attachmentCount = 2u;
			creationParams.attachments = attachments;
			creationParams.dependencies = subpassDeps;
			creationParams.dependencyCount = 3u;
			creationParams.subpasses = subpasses;
			creationParams.subpassCount = subpassCount;

			renderpass = logicalDevice->createGPURenderpass(creationParams);
			if (!renderpass)
				logger->log("Failed to create the render pass!\n", system::ILogger::ELL_ERROR);
		}

		for (uint32_t i = 0u; i < swapchain->getImageCount(); ++i)
		{
			constexpr uint32_t fboAttachmentCount = 2u;
			core::smart_refctd_ptr<video::IGPUImageView> views[fboAttachmentCount] = { nullptr };

			auto swapchainImage = swapchain->getImages().begin()[i];
			{
				video::IGPUImageView::SCreationParams viewParams;
				viewParams.format = swapchainImage->getCreationParameters().format;
				viewParams.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = 1u;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = 1u;
				viewParams.image = std::move(swapchainImage);

				views[0] = logicalDevice->createGPUImageView(std::move(viewParams));
				if (!views[0])
					logger->log("Failed to create swapchain image view %d\n", system::ILogger::ELL_ERROR, i);
			}

			auto depthImage = depthImages[i];
			{
				video::IGPUImageView::SCreationParams viewParams;
				viewParams.format = depthFormat;
				viewParams.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				viewParams.subresourceRange.aspectMask = asset::IImage::EAF_DEPTH_BIT;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = 1u;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = 1u;
				viewParams.image = std::move(depthImage);

				views[1] = logicalDevice->createGPUImageView(std::move(viewParams));
				if (!views[1])
					logger->log("Failed to create depth image view %d\n", system::ILogger::ELL_ERROR, i);
			}

			video::IGPUFramebuffer::SCreationParams creationParams = {};
			creationParams.width = WIN_W;
			creationParams.height = WIN_H;
			creationParams.layers = 1u;
			creationParams.renderpass = renderpass;
			creationParams.flags = static_cast<nbl::video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
			creationParams.attachmentCount = fboAttachmentCount;
			creationParams.attachments = views;
			fbos[i] = logicalDevice->createGPUFramebuffer(std::move(creationParams));

			if (!fbos[i])
				logger->log("Failed to create fbo %d\n", system::ILogger::ELL_ERROR, i);
		}

		// Load in the mesh
		const system::path& archiveFilePath = sharedInputCWD / "sponza.zip";
		const system::path& modelFilePath = sharedInputCWD / "sponza.zip/sponza.obj";
		core::smart_refctd_ptr<asset::ICPUMesh> cpuMesh = nullptr;
		const asset::COBJMetadata* metadataOBJ = nullptr;
		{
			auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
			quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

			auto fileArchive = system->openFileArchive(archiveFilePath);
			// test no alias loading (TODO: fix loading from absolute paths)
			system->mount(std::move(fileArchive));

			asset::IAssetLoader::SAssetLoadParams loadParams;
			loadParams.workingDirectory = sharedInputCWD;
			loadParams.logger = logger.get();
			auto meshesBundle = assetManager->getAsset(modelFilePath.string(), loadParams);
			if (meshesBundle.getContents().empty())
			{
				logger->log("Failed to load the model!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			cpuMesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshesBundle.getContents().begin()[0]);
			metadataOBJ = meshesBundle.getMetadata()->selfCast<const asset::COBJMetadata>();

			quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");
		}
		
		// Setup Camera UBO
		{
			// we can safely assume that all meshbuffers within mesh loaded from OBJ has
			// the same DS1 layout (used for camera-specific data), so we can create just one DS
			const asset::ICPUMeshBuffer* firstMeshBuffer = *cpuMesh->getMeshBuffers().begin();

			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuDSLayout = nullptr;
			{
				auto cpuDSLayout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(CAMERA_DS_NUMBER);

				cpu2gpuParams.beginCommandBuffers();
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuDSLayout, &cpuDSLayout + 1, cpu2gpuParams);
				if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
				{
					logger->log("Failed to convert Camera CPU DS to GPU DS!\n", system::ILogger::ELL_ERROR);
					exit(-1);
				}
				cpu2gpuParams.waitForCreationToComplete();
				gpuDSLayout = (*gpu_array)[0];
			}

			auto dsBindings = gpuDSLayout->getBindings();
			cameraUboBindingNumber = 0u;
			for (const auto& bnd : dsBindings)
			{
				if (bnd.type == asset::EDT_UNIFORM_BUFFER)
				{
					cameraUboBindingNumber = bnd.binding;
					break;
				}
			}

			pipelineMetadata = metadataOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

			size_t uboSize = 0ull;
			for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
				if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUboBindingNumber)
					uboSize = std::max<size_t>(uboSize, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);

			core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool = createDescriptorPool(1u);

			video::IDriverMemoryBacked::SDriverMemoryRequirements ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
			ubomemreq.vulkanReqs.size = uboSize;
			video::IGPUBuffer::SCreationParams gpuuboCreationParams;
			gpuuboCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
			gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
			gpuuboCreationParams.queueFamilyIndexCount = 0u;
			gpuuboCreationParams.queueFamilyIndices = nullptr;
			cameraUbo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams, ubomemreq);
			cameraDS = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuDSLayout));

			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = cameraDS.get();
			write.binding = cameraUboBindingNumber;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_UNIFORM_BUFFER;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = cameraUbo;
				info.buffer.offset = 0ull;
				info.buffer.size = uboSize;
			}
			write.info = &info;
			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		// Convert mesh to GPU objects
		{
			for (size_t i = 0ull; i < cpuMesh->getMeshBuffers().size(); ++i)
			{
				auto& meshBuffer = cpuMesh->getMeshBuffers().begin()[i];

				// Todo(achal): Can get rid of this probably after
				// https://github.com/Devsh-Graphics-Programming/Nabla/pull/160#discussion_r747185441
				for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
					meshBuffer->getPipeline()->getBlendParams().blendParams[i].attachmentEnabled = (i == 0ull);

				meshBuffer->getPipeline()->getRasterizationParams().frontFaceIsCCW = false;
			}

			cpu2gpuParams.beginCommandBuffers();
			asset::ICPUMesh* meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());
			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
			if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			{
				logger->log("Failed to convert mesh to GPU objects!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
			cpu2gpuParams.waitForCreationToComplete();
			gpuMesh = (*gpu_array)[0];
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &bakedCommandBuffer);
		{
			core::unordered_map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelineCache;
			core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines(gpuMesh->getMeshBuffers().size());
			for (size_t i = 0ull; i < graphicsPipelines.size(); ++i)
			{
				const video::IGPURenderpassIndependentPipeline* renderpassIndep = gpuMesh->getMeshBuffers()[i]->getPipeline();
				video::IGPURenderpassIndependentPipeline* renderpassIndep_mutable = const_cast<video::IGPURenderpassIndependentPipeline*>(renderpassIndep);
				auto& rasterizationParams_mutable = const_cast<asset::SRasterizationParams&>(renderpassIndep_mutable->getRasterizationParams());
				rasterizationParams_mutable.depthCompareOp = asset::ECO_GREATER_OR_EQUAL;

				auto foundPpln = graphicsPipelineCache.find(renderpassIndep);
				if (foundPpln == graphicsPipelineCache.end())
				{
					video::IGPUGraphicsPipeline::SCreationParams params;
					params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
					params.renderpass = core::smart_refctd_ptr(renderpass);
					params.subpassIx = lightingPassIndex;
					foundPpln = graphicsPipelineCache.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
				}
				graphicsPipelines[i] = foundPpln->second;
			}
			if (!bakeSecondaryCommandBufferForSubpass(lightingPassIndex, bakedCommandBuffer.get(), graphicsPipelines))
			{
				logger->log("Failed to create lighting pass command buffer!", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &zPrepassCommandBuffer);
		{
			core::unordered_map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelineCache;
			core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines(gpuMesh->getMeshBuffers().size());
			for (size_t i = 0ull; i < graphicsPipelines.size(); ++i)
			{
				const video::IGPURenderpassIndependentPipeline* renderpassIndep = gpuMesh->getMeshBuffers()[i]->getPipeline();
				video::IGPURenderpassIndependentPipeline* renderpassIndep_mutable = const_cast<video::IGPURenderpassIndependentPipeline*>(renderpassIndep);
				auto foundPpln = graphicsPipelineCache.find(renderpassIndep);
				if (foundPpln == graphicsPipelineCache.end())
				{
					// There is no other way for me currently to "disable" a shader stage
					// from an already existing renderpass independent pipeline. This
					// needs to be done for the z pre pass
					renderpassIndep_mutable->setShaderAtStage(asset::IShader::ESS_FRAGMENT, nullptr);

					video::IGPUGraphicsPipeline::SCreationParams params;
					params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
					params.renderpass = core::smart_refctd_ptr(renderpass);
					params.subpassIx = zPrepassIndex;
					foundPpln = graphicsPipelineCache.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
				}
				graphicsPipelines[i] = foundPpln->second;
			}
			if (!bakeSecondaryCommandBufferForSubpass(zPrepassIndex, zPrepassCommandBuffer.get(), graphicsPipelines))
			{
				logger->log("Failed to create depth pre-pass command buffer!", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		core::vectorSIMDf cameraPosition(-157.229813, 169.800446, -19.696722, 0.000000);
		core::vectorSIMDf cameraTarget(-387.548462, 198.927414, -26.500174, 1.000000);

		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, 4000.f);
		camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 10.f, 1.f);

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

			core::vector<uint8_t> uboData(cameraUbo->getSize());
			for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
			{
				if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUboBindingNumber)
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
			commandBuffer->updateBuffer(cameraUbo.get(), 0ull, cameraUbo->getSize(), uboData.data());
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

			video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
			{
				VkRect2D area;
				area.offset = { 0,0 };
				area.extent = { WIN_W, WIN_H };
				asset::SClearValue clear[2] = {};
				clear[0].color.float32[0] = 1.f;
				clear[0].color.float32[1] = 0.f;
				clear[0].color.float32[2] = 1.f;
				clear[0].color.float32[3] = 1.f;
				clear[1].depthStencil.depth = 0.f;

				beginInfo.clearValueCount = 2u;
				beginInfo.framebuffer = fbos[acquiredNextFBO];
				beginInfo.renderpass = renderpass;
				beginInfo.renderArea = area;
				beginInfo.clearValues = clear;
			}

			commandBuffer->beginRenderPass(&beginInfo, asset::ESC_SECONDARY_COMMAND_BUFFERS);
			commandBuffer->executeCommands(1u, &zPrepassCommandBuffer.get());
			commandBuffer->nextSubpass(asset::ESC_SECONDARY_COMMAND_BUFFERS);
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

private:
	bool bakeSecondaryCommandBufferForSubpass(const uint32_t subpass, video::IGPUCommandBuffer* cmdbuf, const core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>>& graphicsPipelines)
	{
		assert(gpuMesh->getMeshBuffers().size() == graphicsPipelines.size());

		video::IGPUCommandBuffer::SInheritanceInfo inheritanceInfo = {};
		inheritanceInfo.renderpass = renderpass;
		inheritanceInfo.subpass = subpass;
		// inheritanceInfo.framebuffer = ; // might be good to have it

		cmdbuf->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT | video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT, &inheritanceInfo);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		cmdbuf->setViewport(0u, 1u, &viewport);

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W, WIN_H };
		cmdbuf->setScissor(0u, 1u, &scissor);

		{
			const uint32_t drawCallCount = gpuMesh->getMeshBuffers().size();

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

			video::CSubpassKiln subpassKiln;

			auto drawCallData = new asset::DrawElementsIndirectCommand_t[drawCallCount];
			{
				auto drawIndexIt = allocation.multiDrawCommandRangeByteOffsets;
				auto drawCallDataIt = drawCallData;

				for (size_t i = 0ull; i < gpuMesh->getMeshBuffers().size(); ++i)
				{
					const auto& mb = gpuMesh->getMeshBuffers().begin()[i];
					auto& drawcall = subpassKiln.getDrawcallMetadataVector().emplace_back();

					// push constants
					memcpy(drawcall.pushConstantData, mb->getPushConstantsDataPtr(), video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
					// graphics pipeline
					drawcall.pipeline = graphicsPipelines[i];
					// descriptor sets
					drawcall.descriptorSets[1] = cameraDS;
					drawcall.descriptorSets[3] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(mb->getAttachedDescriptorSet());
					// vertex buffers
					std::copy_n(mb->getVertexBufferBindings(), video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT, drawcall.vertexBufferBindings);
					// index buffer
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

				auto upQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
				core::smart_refctd_ptr<video::IGPUCommandBuffer> tferCmdBuf;
				logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &tferCmdBuf);

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
					if (ppHandler->transferProperties(
						utilities->getDefaultUpStreamingBuffer(), tferCmdBuf.get(), fence.get(), upQueue,
						scratch, pRequest, 1u, waitSemaphoreCount, waitSemaphores, waitStages,
						logger.get(), std::chrono::high_resolution_clock::time_point::max() // wait forever if necessary, need initialization to finish
					))
						return false;
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

			subpassKiln.bake(cmdbuf, renderpass.get(), subpass, drawAllocator->getDrawCommandMemoryBlock().buffer.get(), nullptr);
		}
		cmdbuf->end();

		return true;
	}

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
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
	core::smart_refctd_ptr<video::IGPUFramebuffer> fbos[CommonAPI::InitOutput::MaxSwapChainImageCount] = { nullptr };
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
	core::smart_refctd_ptr<video::IGPUCommandBuffer> zPrepassCommandBuffer;

	const asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;

	core::smart_refctd_ptr<video::IGPUBuffer> cameraUbo;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> cameraDS;
	uint32_t cameraUboBindingNumber;

	core::smart_refctd_ptr<video::IGPUMesh> gpuMesh;

	video::CDumbPresentationOracle oracle;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());
};

NBL_COMMON_API_MAIN(ClusteredRenderingSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }