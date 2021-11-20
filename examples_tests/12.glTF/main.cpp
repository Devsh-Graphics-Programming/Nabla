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

#include "nbl/asset/metadata/CGLTFMetadata.h"
#include "nbl/scene/CSkinInstanceCache.h"

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;

class GLTFApp : public ApplicationBase
{
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_H = 720;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t SC_IMG_COUNT = 3u;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	public:
        void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
        {
            window = std::move(wnd);
        }
        void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
        {
            system = std::move(s);
        }
        nbl::ui::IWindow* getWindow() override
        {
            return window.get();
        }

		APP_CONSTRUCTOR(GLTFApp)
		void onAppInitialized_impl() override
		{
			initOutput.window = core::smart_refctd_ptr(window);

			CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "glTF", asset::EF_D32_SFLOAT);
			window = std::move(initOutput.window);
			gl = std::move(initOutput.apiConnection);
			surface = std::move(initOutput.surface);
			gpuPhysicalDevice = std::move(initOutput.physicalDevice);
			logicalDevice = std::move(initOutput.logicalDevice);
			queues = std::move(initOutput.queues);
			swapchain = std::move(initOutput.swapchain);
			renderpass = std::move(initOutput.renderpass);
			fbos = std::move(initOutput.fbo);
			commandPool = std::move(initOutput.commandPool);
			assetManager = std::move(initOutput.assetManager);
			logger = std::move(initOutput.logger);
			inputSystem = std::move(initOutput.inputSystem);
			system = std::move(initOutput.system);
			windowCallback = std::move(initOutput.windowCb);
			cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
			utilities = std::move(initOutput.utilities);

			transferUpQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
			
			auto transformTreeManager = scene::ITransformTreeManager::create(utilities.get(),transferUpQueue);


			auto gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
			auto gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

			nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
			{
				cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
				cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
			}

			auto cpu2gpuWaitForFences = [&]() -> void
			{
				video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
				while (waitStatus != video::IGPUFence::ES_SUCCESS)
				{
					waitStatus = logicalDevice->waitForFences(1u, &gpuTransferFence.get(), false, 99999999ull);
					if (waitStatus == video::IGPUFence::ES_ERROR)
						assert(false);
					else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
						break;
				}

				waitStatus = video::IGPUFence::ES_NOT_READY;
				while (waitStatus != video::IGPUFence::ES_SUCCESS)
				{
					waitStatus = logicalDevice->waitForFences(1u, &gpuComputeFence.get(), false, 99999999ull);
					if (waitStatus == video::IGPUFence::ES_ERROR)
						assert(false);
					else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
						break;
				}
			};

			auto createDescriptorPool = [&](const uint32_t amount, const E_DESCRIPTOR_TYPE type) // TODO: review
			{
				constexpr uint32_t maxItemCount = 256u;
				{
					nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
					poolSize.count = amount;
					poolSize.type = type;
					return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
				}
			};
			
			core::set<asset::ICPUDescriptorSetLayout*> cpuDS1Layouts;
			struct LoadedGLTF
			{
				const asset::CGLTFMetadata* meta = nullptr;
				core::vector<core::smart_refctd_ptr<asset::ICPUMeshBuffer>> meshbuffers;
			};
			core::vector<LoadedGLTF> models;

			auto loadRiggedGLTF = [&](const system::path& filename) -> void
			{
				auto resourcePath = sharedInputCWD / "../../3rdparty/glTFSampleModels/2.0/"; // TODO: fix up for Android

				asset::IAssetLoader::SAssetLoadParams loadingParams = {};
				auto meshes_bundle = assetManager->getAsset((resourcePath/filename).string(),loadingParams);
				auto contents = meshes_bundle.getContents();
				if (contents.empty())
					return;

				LoadedGLTF model;
				model.meta = meshes_bundle.getMetadata()->selfCast<asset::CGLTFMetadata>();

				for (const auto& asset : contents)
				{
					auto mesh = IAsset::castDown<ICPUMesh>(asset.get());
					for (const auto& meshbuffer : mesh->getMeshBuffers())
					{
						cpuDS1Layouts.insert(meshbuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1));
						model.meshbuffers.emplace_back(meshbuffer);
					}
				}
				models.push_back(std::move(model));
			};
			loadRiggedGLTF("AnimatedTriangle/glTF/AnimatedTriangle.gltf");
// TODO: @AnastaZIuk this one crashes the loader!
//			loadRiggedGLTF("IridescentDishWithOlives/glTF/IridescentDishWithOlives.gltf");
			loadRiggedGLTF("RiggedFigure/glTF/RiggedFigure.gltf"); 
			loadRiggedGLTF("RiggedSimple/glTF/RiggedSimple.gltf");
			loadRiggedGLTF("SimpleSkin/glTF/SimpleSkin.gltf"); // see it @devshgraphicsprogramming
			// TODO: support playback of keyframe animations to nodes which don't have skinning
			loadRiggedGLTF("AnimatedCube/glTF/AnimatedCube.gltf");
			loadRiggedGLTF("BoxAnimated/glTF/BoxAnimated.gltf");
			loadRiggedGLTF("InterpolationTest/glTF/InterpolationTest.gltf");
			// TODO: support node without skeleton or animations
// TODO: @AnastaZIuk this one crashes the loader!
//			loadRiggedGLTF("FlightHelmet/glTF/FlightHelmet.gltf"); 
			// TODO: nightmare case, handle in far future
			//loadRiggedGLTF("RecursiveSkeletons/glTF/RecursiveSkeletons.gltf");
			

#if 0

			// Transform Tree
			constexpr uint32_t MaxNodeCount = 128u<<10u; // get ready for many many nodes
			auto transformTree = scene::ITransformTreeWithNormalMatrices::create(logicalDevice.get(),MaxNodeCount);

			auto ppHandler = utilities->getDefaultPropertyPoolHandler();
			// transfer cmdbuf and fences
			nbl::core::smart_refctd_ptr<nbl::video::IGPUFence> xferFence;
			nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> xferCmdbuf;
			{
				xferFence = logicalDevice->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));
				logicalDevice->createCommandBuffers(commandPool.get(),nbl::video::IGPUCommandBuffer::EL_PRIMARY,1u,&xferCmdbuf);
				xferCmdbuf->begin(0);
			}
			auto xferQueue = logicalDevice->getQueue(xferCmdbuf->getQueueFamilyIndex(),0u);
			asset::SBufferBinding<video::IGPUBuffer> xferScratch;
			{
				video::IGPUBuffer::SCreationParams scratchParams = {};
				scratchParams.canUpdateSubRange = true;
				scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT)|video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				xferScratch = {0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,ppHandler->getMaxScratchSize())};
				xferScratch.buffer->setObjectDebugName("PropertyPoolHandler Scratch Buffer");
			}

			// add skeleton instances to transform tree
			core::vector<const asset::ICPUSkeleton*> skeletons;
			core::vector<uint32_t> skeletonInstanceCounts;
			for (auto& model : models)
			{
				const asset::ICPUSkeleton* skeleton = nullptr;
				for (const auto& meshbuffer : model.meshbuffers)
				{
					const auto mesh_skel = meshbuffer->getSkeleton();
					if (!mesh_skel)
					{
						logger->log("A meshbuffer of a model with metadata ptr % p doesn't have a meshbuffer.",system::ILogger::ELL_WARNING,model.meta);
						continue;
					}
					if (!skeleton)
						skeleton = mesh_skel;
					else if (mesh_skel!=skeleton)
						logger->log("A meshbuffer of a model with metadata ptr %p has a different skeleton, possible loader bug.",system::ILogger::ELL_WARNING,model.meta);
				}
				if (skeleton)
				{
					skeletons.push_back(skeleton);
					skeletonInstanceCounts.push_back(2u); // TODO: rand
				}
			}
			// allocate skeleton nodes in TT
			core::vector<scene::ITransformTree::node_t> allSkeletonNodes;
			{
				scene::ITransformTreeManager::SkeletonAllocationRequest skeletonAllocationRequest;
				skeletonAllocationRequest.tree = transformTree.get();
				skeletonAllocationRequest.cmdbuf = xferCmdbuf.get();
				skeletonAllocationRequest.fence = xferFence.get();
				skeletonAllocationRequest.scratch = xferScratch;
				skeletonAllocationRequest.upBuff = utilities->getDefaultUpStreamingBuffer();
				skeletonAllocationRequest.poolHandler = ppHandler;
				skeletonAllocationRequest.queue = xferQueue;
				skeletonAllocationRequest.logger = logger.get();
				skeletonAllocationRequest.skeletons = {skeletons.data(),skeletons.data()+skeletons.size()};
				skeletonAllocationRequest.instanceCounts = skeletonInstanceCounts.data();
				skeletonAllocationRequest.skeletonInstanceParents = nullptr; // no parent so we can instance/duplicate the skeleton freely

				auto stagingReqs = skeletonAllocationRequest.computeStagingRequirements();
				allSkeletonNodes.resize(stagingReqs.nodeCount,scene::ITransformTree::invalid_node);
				core::vector<scene::ITransformTree::node_t> parentScratch(stagingReqs.nodeCount);
				core::vector<scene::ITransformTree::relative_transform_t> transformScratch(stagingReqs.transformScratchCount);

				skeletonAllocationRequest.outNodes = allSkeletonNodes.data();
				skeletonAllocationRequest.parentScratch = parentScratch.data();
				skeletonAllocationRequest.transformScratch = transformScratch.data();

				uint32_t waitSemaphoreCount = 0u;
				video::IGPUSemaphore*const* waitSempahores = nullptr;
				const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
				transformTreeManager->addSkeletonNodes(skeletonAllocationRequest,waitSemaphoreCount,waitSempahores,waitStages);
			}
			/*
			struct SkeletonInstance
			{
			public:
				SkeletonInstance(const asset::ICPUSkeleton* _skeleton, const uint32_t _instanceCount)
					: m_instanceNodes(_skeleton->getJointCount()* _instanceCount, scene::ITransformTree::invalid_node),
					m_skeleton(_skeleton), m_instanceCount(_instanceCount), m_jointCount(_skeleton->getJointCount())
				{
				}

				inline scene::ITransformTree::node_t* getInstanceNodes(const uint32_t instanceID)
				{
					if (instanceID < m_instanceCount)
						return m_instanceNodes.data() + m_jointCount * instanceID;
					return nullptr;
				}

			private:
				core::vector<scene::ITransformTree::node_t> m_instanceNodes;
				const asset::ICPUSkeleton* m_skeleton;
				uint32_t m_instanceCount;
				uint32_t m_jointCount;
			};
			using skeleton_instance_nodes_t = core::vector<scene::ITransformTree::node_t>;
			core::map<asset::ICPUSkeleton*, skeleton_instance_nodes_t> skeletonInstances;
			*/

			// transfer submit
			{
				xferCmdbuf->end();
				{
					video::IGPUQueue::SSubmitInfo submit;
					submit.commandBufferCount = 1u;
					submit.commandBuffers = &xferCmdbuf.get();
					xferQueue->submit(1u,&submit,xferFence.get());
				}
				logicalDevice->blockForFences(1u,&xferFence.get());
				logicalDevice->resetFences(1u,&xferFence.get());
			}

#if 0

			/*
				Property Buffers for skinning
			*/
			asset::SBufferRange<video::IGPUBuffer> debugAABBs;
			{
				video::IGPUBuffer::SCreationParams params = {};
				params.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				debugAABBs.buffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(params, sizeof(core::aabbox3df) * MaxNodeCount);
			}

			/*
				We can safely assume that all meshes' mesh buffers loaded from glTF has the same DS1 layout
				used for camera-specific data, so we can create just one DS.
			*/

			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuDescriptorSet1Layout;
			{
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuDescriptorSetLayout1, &cpuDescriptorSetLayout1 + 1, cpu2gpuParams);
				if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				//cpu2gpuWaitForFences(); still doesn't work?
				gpuDescriptorSet1Layout = (*gpu_array)[0];
			}

			auto uboMemoryReqs = logicalDevice->getDeviceLocalGPUMemoryReqs();
			uboMemoryReqs.vulkanReqs.size = sizeof(SBasicViewParameters);

			video::IGPUBuffer::SCreationParams uboBufferCreationParams;
			uboBufferCreationParams.canUpdateSubRange = true;
			uboBufferCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT;

			gpuubo = logicalDevice->createGPUBufferOnDedMem(uboBufferCreationParams, uboMemoryReqs);
			gpuUboDescriptorPool = createDescriptorPool(1u, EDT_UNIFORM_BUFFER);

			gpuDescriptorSet1 = logicalDevice->createGPUDescriptorSet(gpuUboDescriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSet1Layout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet write;
				write.dstSet = gpuDescriptorSet1.get();
				write.binding = 0;
				write.count = 1u;
				write.arrayElement = 0u;
				write.descriptorType = asset::EDT_UNIFORM_BUFFER;
				video::IGPUDescriptorSet::SDescriptorInfo info;
				{
					info.desc = gpuubo;
					info.buffer.offset = 0ull;
					info.buffer.size = sizeof(SBasicViewParameters);
				}
				write.info = &info;
				logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
			}

			/*
				TODO: MORE DS3s, use metadata to fetch translations, track it and set up in the graphicsData struct
			*/

			for (auto* asset = meshes_bundle.getContents().begin(); asset != meshes_bundle.getContents().end(); ++asset)
			{
				auto& graphicsDataMesh = graphicsData.meshes.emplace_back();

				auto cpuMesh = core::smart_refctd_ptr_static_cast<ICPUMesh>(*asset);
				{
					for (size_t i = 0; i < cpuMesh->getMeshBuffers().size(); ++i)
					{
						auto& graphicsResources = graphicsDataMesh.resources.emplace_back();
						auto* glTFMetaPushConstants = glTFMeta->getAssetSpecificMetadata(cpuMesh->getMeshBufferVector()[i]->getPipeline());
						graphicsResources.pipelineMetadata = glTFMetaPushConstants;
					}
				}

				core::smart_refctd_ptr<video::IGPUMesh> gpuMesh;
				{
					auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuMesh.get(), &cpuMesh.get() + 1, cpu2gpuParams);
					if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
						assert(false);

					//cpu2gpuWaitForFences(); still doesn't work?
					gpuMesh = (*gpu_array)[0];
				}

				using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
				std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;

				for (size_t i = 0; i < gpuMesh->getMeshBuffers().size(); ++i)
				{
					auto* gpuMeshBuffer = graphicsDataMesh.resources[i].gpuMeshBuffer = (gpuMesh->getMeshBufferIterator() + i)->get();
					auto* gpuRenderpassIndependentPipeline = graphicsDataMesh.resources[i].gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();

					const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuRenderpassIndependentPipeline);
					const auto alreadyCreated = gpuPipelines.contains(adress);
					{
						if (!alreadyCreated)
						{
							nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
							graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuRenderpassIndependentPipeline));
							graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

							gpuPipelines[adress] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
						}

						graphicsDataMesh.resources[i].gpuGraphicsPipeline = gpuPipelines[adress];
					}
				}
			}
#endif
			core::vectorSIMDf cameraPosition(-0.5, 0, 0);
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.01f, 10000.0f);
			camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 0.04f, 1.f);
			auto lastTime = std::chrono::system_clock::now();

			logicalDevice->createCommandBuffers(commandPool.get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,commandBuffers);

			//
			oracle.reportBeginFrameRecord();
			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				imageAcquire[i] = logicalDevice->createSemaphore();
				renderFinished[i] = logicalDevice->createSemaphore();
			}

#endif
		}

		void onAppTerminated_impl() override
		{
			const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
			auto gpuSourceImageView = fboCreationParams.attachments[0];

			bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], renderFinished[resourceIx].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
			assert(status);
		}

		void workLoopBody() override
		{
			exit(69);

			++resourceIx;
			if (resourceIx >= FRAMES_IN_FLIGHT)
				resourceIx = 0;

			auto& commandBuffer = commandBuffers[resourceIx];
			auto& fence = frameComplete[resourceIx];

			if (fence)
				logicalDevice->blockForFences(1u, &fence.get());
			else
				fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

			commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
			commandBuffer->begin(0);

			const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

			inputSystem->getDefaultMouse(&mouse);
			inputSystem->getDefaultKeyboard(&keyboard);

			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
			camera.endInputProcessing(nextPresentationTimestamp);

			const auto& viewMatrix = camera.getViewMatrix();
			const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

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

			commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);



#if 0
			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
			modelMatrix.setRotation(quaternion(0, 0, 0));

			core::matrix3x4SIMD modelViewMatrix = core::concatenateBFollowedByA(viewMatrix, modelMatrix);
			core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

			core::matrix3x4SIMD normalMatrix;
			modelViewMatrix.getSub3x3InverseTranspose(normalMatrix);

			/*
				Camera data is shared between all meshes
			*/

			SBasicViewParameters uboData;
			memcpy(uboData.MVP, modelViewProjectionMatrix.pointer(), sizeof(uboData.MVP));
			memcpy(uboData.MV, modelViewMatrix.pointer(), sizeof(uboData.MV));
			memcpy(uboData.NormalMat, normalMatrix.pointer(), sizeof(uboData.NormalMat));

			commandBuffer->updateBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

			for (auto& gpuMeshData : graphicsData.meshes)
			{
				for (auto& graphicsResource : gpuMeshData.resources)
				{
					const auto* gpuMeshBuffer = graphicsResource.gpuMeshBuffer;
					auto gpuGraphicsPipeline = core::smart_refctd_ptr(graphicsResource.gpuGraphicsPipeline);

					const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = graphicsResource.gpuRenderpassIndependentPipeline;
					const video::IGPUDescriptorSet* gpuDescriptorSet3 = gpuMeshBuffer->getAttachedDescriptorSet();

					commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
					commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), nullptr);

					if (gpuDescriptorSet3)
						commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuDescriptorSet3, nullptr);

					static_assert(sizeof(asset::CGLTFPipelineMetadata::SGLTFMaterialParameters) <= video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
					commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, sizeof(asset::CGLTFPipelineMetadata::SGLTFMaterialParameters), &graphicsResource.pipelineMetadata->m_materialParams);

					commandBuffer->drawMeshBuffer(gpuMeshBuffer);
				}
			}
#endif



			commandBuffer->endRenderPass();
			commandBuffer->end();

			CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
			CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
		}

		bool keepRunning() override
		{
			return windowCallback->isWindowOpen();
		}

	private:
		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCallback;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;

		nbl::video::IGPUQueue* transferUpQueue = nullptr;

		Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());
		CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
		CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
		core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

		video::CDumbPresentationOracle oracle;

		_NBL_STATIC_INLINE_CONSTEXPR uint64_t MAX_TIMEOUT = 99999999999999ull;
		uint32_t acquiredNextFBO = {};
		int32_t resourceIx = -1;
};

NBL_COMMON_API_MAIN(GLTFApp)