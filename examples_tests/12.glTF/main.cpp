// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include <iostream>
#include <cstdio>
#include <random>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

// TODO: move
#include "nbl/asset/metadata/CGLTFMetadata.h"
#include "nbl/scene/CSkinInstanceCache.h"
#include "nbl/scene/ISkinInstanceCacheManager.h"

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
		nbl::asset::E_FORMAT getDepthFormat() override
		{
			return nbl::asset::EF_D32_SFLOAT;
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
			
			transformTreeManager = scene::ITransformTreeManager::create(utilities.get(),transferUpQueue);
			ttDebugDrawPipeline = transformTreeManager->createDebugPipeline<scene::ITransformTreeWithNormalMatrices>(core::smart_refctd_ptr(renderpass));

			sicManager = scene::ISkinInstanceCacheManager::create(utilities.get(),transferUpQueue);

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
//			loadRiggedGLTF("AnimatedTriangle/glTF/AnimatedTriangle.gltf");
// TODO: @AnastaZIuk this one crashes the loader!
//			loadRiggedGLTF("IridescentDishWithOlives/glTF/IridescentDishWithOlives.gltf");
			loadRiggedGLTF("RiggedFigure/glTF/RiggedFigure.gltf"); 
//			loadRiggedGLTF("RiggedSimple/glTF/RiggedSimple.gltf");
//			loadRiggedGLTF("SimpleSkin/glTF/SimpleSkin.gltf");
			// TODO: support playback of keyframe animations to nodes which don't have skinning
//			loadRiggedGLTF("AnimatedCube/glTF/AnimatedCube.gltf");
//			loadRiggedGLTF("BoxAnimated/glTF/BoxAnimated.gltf");
//			loadRiggedGLTF("InterpolationTest/glTF/InterpolationTest.gltf");
			// TODO: support node without skeleton or animations
// TODO: @AnastaZIuk this one crashes the loader!
//			loadRiggedGLTF("FlightHelmet/glTF/FlightHelmet.gltf"); 
			// TODO: nightmare case, handle in far future
			//loadRiggedGLTF("RecursiveSkeletons/glTF/RecursiveSkeletons.gltf");
			

			// Transform Tree
			constexpr uint32_t MaxNodeCount = 2u<<10u; // get ready for many many nodes
			transformTree = scene::ITransformTreeWithNormalMatrices::create(logicalDevice.get(),MaxNodeCount);

			// Skinning Cache
			{
				scene::ISkinInstanceCache::ImplicitCreationParameters params;
				params.device = logicalDevice.get();
				params.associatedTransformTree = transformTree;
				params.inverseBindPoseCapacity = 1u<<16u;
				params.jointCapacity = 4u<<10u;
				skinInstanceCache = scene::CSkinInstanceCache<>::create(std::move(params));
			}

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

			std::mt19937 mt(0x45454545u);
			// three levels of instancing:
			// - instancing within a model (mesh instances)
			// - skeleton instancing (incl copying currently playing animations)
			// - model instancing TODO
			core::vector<uint32_t> modelInstanceCounts; // TODO rename
			// add skeleton instances to transform tree (abuse the value as first an instance count, then to store the offset to the node handles TODO)
			core::unordered_map<const asset::ICPUSkeleton*,uint32_t> skeletonInstances;
			for (const auto& model : models)
			{
				const auto instanceCount = modelInstanceCounts.emplace_back() = std::uniform_int_distribution<uint32_t>(1,5)(mt);
				for (const auto& skeleton : model.meta->skeletons)
				{
					auto found = skeletonInstances.find(skeleton.get());
					if (found!=skeletonInstances.end())
						found->second += instanceCount;
					else
						skeletonInstances.insert({skeleton.get(),instanceCount});
				}
			}
			// allocate skeleton nodes in TT
			core::vector<scene::ITransformTree::node_t> allSkeletonNodes;
			{
				core::vector<const ICPUSkeleton*> skeletons; skeletons.reserve(skeletonInstances.size());
				core::vector<uint32_t> skeletonInstanceCounts; skeletonInstanceCounts.reserve(skeletonInstances.size());
				for (const auto& pair : skeletonInstances)
				{
					skeletons.push_back(pair.first);
					skeletonInstanceCounts.push_back(pair.second);
				}

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
			struct InverseBindPoseRangeHash
			{
				inline size_t operator()(const asset::SBufferRange<const asset::ICPUBuffer>& inverseBindPoseRange) const
				{
					return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&inverseBindPoseRange),sizeof(inverseBindPoseRange)));
				}
			};
			core::unordered_map<asset::SBufferRange<const asset::ICPUBuffer>,uint32_t,InverseBindPoseRangeHash> inverseBindPoseRanges;
			struct Skin
			{
				const ICPUSkeleton* skeleton;
				asset::SBufferBinding<const asset::ICPUBuffer> skinTranslationTable;
				asset::SBufferBinding<const asset::ICPUBuffer> inverseBindPoses;

				inline bool operator==(const Skin& other) const
				{
					return skeleton==other.skeleton && skinTranslationTable==other.skinTranslationTable && inverseBindPoses==other.inverseBindPoses;
				}
			};
			struct SkinHash
			{
				inline size_t operator()(const Skin& skin) const
				{
					return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&skin),sizeof(skin)));
				}
			};
			core::unordered_map<Skin,uint32_t,SkinHash> skinInstances;
			// pick a scene and flag all skin instances
			for (auto i=0u; i<models.size(); i++)
			{
				const auto modelInstanceCount = modelInstanceCounts[i];
				const auto* meta = models[i].meta;
				// pick a scene
				const auto& scenes = meta->scenes;
				const auto sceneID = meta->defaultSceneID<scenes.size() ? meta->defaultSceneID:0u;
				const auto& scene = scenes[sceneID];
				for (const auto& instanceID : scene.instanceIDs)
				{
					const auto& instance = meta->instances[instanceID];
					//instance.attachedToNode 
					for (const auto& meshbuffer : instance.mesh->getMeshBuffers())
					{
						asset::SBufferRange<const asset::ICPUBuffer> inverseBindPoseRange;
						inverseBindPoseRange.offset = meshbuffer->getInverseBindPoseBufferBinding().offset;
						inverseBindPoseRange.size = sizeof(scene::ISkinInstanceCache::inverse_bind_pose_t)*meshbuffer->getJointCount();
						inverseBindPoseRange.buffer = meshbuffer->getInverseBindPoseBufferBinding().buffer;
						if (inverseBindPoseRange.buffer)
						{
							auto foundIBPR = inverseBindPoseRanges.find(inverseBindPoseRange);
							if (foundIBPR==inverseBindPoseRanges.end())
								inverseBindPoseRanges.insert({std::move(inverseBindPoseRange),video::IPropertyPool::invalid});
						}

						Skin skin = {instance.skeleton,instance.skinTranslationTable,meshbuffer->getInverseBindPoseBufferBinding()};
						auto foundSkin = skinInstances.find(skin);
						if (foundSkin!=skinInstances.end())
							foundSkin->second += modelInstanceCount;
						else
							skinInstances.insert({std::move(skin),modelInstanceCount});
					}
				}
			}
			// allocate an inverse bind pose for every inverseBindPose
			{
				// temporary debug
				for (const auto& pair : inverseBindPoseRanges)
				{
					const auto& ibpr = pair.first;
					const uint8_t* ptr = reinterpret_cast<const uint8_t*>(ibpr.buffer->getPointer())+ibpr.offset;
					auto inverseBindPoseIt = reinterpret_cast<const scene::ISkinInstanceCache::inverse_bind_pose_t*>(ptr);
					auto end = reinterpret_cast<const scene::ISkinInstanceCache::inverse_bind_pose_t*>(ptr+ibpr.size);
					while (inverseBindPoseIt!=end)
					{
						printf("\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
							inverseBindPoseIt->rows[0].x,inverseBindPoseIt->rows[0].y,inverseBindPoseIt->rows[0].z,inverseBindPoseIt->rows[0].w,
							inverseBindPoseIt->rows[1].x,inverseBindPoseIt->rows[1].y,inverseBindPoseIt->rows[1].z,inverseBindPoseIt->rows[1].w,
							inverseBindPoseIt->rows[2].x,inverseBindPoseIt->rows[2].y,inverseBindPoseIt->rows[2].z,inverseBindPoseIt->rows[2].w
						);
						inverseBindPoseIt++;
					}
				}
			}
			// allocate a skin cache entry for every skin instance
			{
			}
			//temp
			{
				IGPUBuffer::SCreationParams params = {};
				params.usage = core::bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT)|IGPUBuffer::EUF_VERTEX_BUFFER_BIT;

				totalJointCount = allSkeletonNodes.size();
				allSkeletonNodes.insert(allSkeletonNodes.begin(),totalJointCount);
				allSkeletonNodesBinding = {0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(scene::ITransformTree::node_t)*allSkeletonNodes.size(),allSkeletonNodes.data())};
				iotaBinding = {0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(uint32_t)*totalJointCount)};

				ttmDescriptorSets = transformTreeManager->createAllDescriptorSets(logicalDevice.get());
				transformTreeManager->updateRecomputeGlobalTransformsDescriptorSet(logicalDevice.get(),ttmDescriptorSets.recomputeGlobal.get(),SBufferBinding(allSkeletonNodesBinding));
				
				const core::aabbox3df defaultAABB(-0.01f,-0.01f,-0.01f,0.01f,0.01f,0.01f);
				core::vector<CompressedAABB> tmp(MaxNodeCount,defaultAABB);
				transformTreeManager->updateDebugDrawDescriptorSet(
					logicalDevice.get(),ttmDescriptorSets.debugDraw.get(),{0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(
						transferUpQueue,sizeof(CompressedAABB)*MaxNodeCount,tmp.data()
					)}
				);
			}
			// TODO: skin instance cache finish code
			// TODO: skin instance cache manager update shader
			// TODO: skin instance cache debug draw shader

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
				//logicalDevice->resetFences(1u,&xferFence.get());
			}

			// TODO: vertex shader skinning override
			// TODO: create the IGPUMeshes
			// TODO: draw them all instanced
			// TODO: use the LoD system!?

			//Animation before or after draw?
#if 0
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

			// Update node transforms 
			{
				// buffers to barrier w.r.t. updates
				video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[scene::ITransformTreeManager::SBarrierSuggestion::MaxBufferCount];
				auto setBufferBarrier = [&barriers,commandBuffer](const uint32_t ix, const asset::SBufferRange<video::IGPUBuffer>& range, const asset::SMemoryBarrier& barrier)
				{
					barriers[ix].barrier = barrier;
					barriers[ix].dstQueueFamilyIndex = barriers[ix].srcQueueFamilyIndex = commandBuffer->getQueueFamilyIndex();
					barriers[ix].buffer = range.buffer;
					barriers[ix].offset = range.offset;
					barriers[ix].size = range.size;
				};

				const core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> renderingStages = asset::EPSF_VERTEX_SHADER_BIT;
				 
				scene::ITransformTreeManager::BaseParams baseParams;
				baseParams.cmdbuf = commandBuffer.get();
				baseParams.tree = transformTree.get();
				baseParams.logger = initOutput.logger.get();

				// compilers are too dumb to figure out const correctness (there's also a TODO in `core::smart_refctd_ptr`)
				const scene::ITransformTree* ptt = transformTree.get();
				const video::IPropertyPool* node_pp = ptt->getNodePropertyPool();
				/* tODO
				{
					auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_PRE_RELATIVE_TFORM_UPDATE);
					sugg.srcStageMask |= asset::EPSF_TRANSFER_BIT; // barrier after buffer upload, before TTM updates (so TTM update CS gets properly written data)
					sugg.requestRanges.srcAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
					sugg.modificationRequests.srcAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
					uint32_t barrierCount = 0u;
					setBufferBarrier(barrierCount++, { 0ull,modRangesBuf->getSize(),modRangesBuf }, sugg.requestRanges);
					setBufferBarrier(barrierCount++, { 0ull,relTformModsBuf->getSize(),relTformModsBuf }, sugg.modificationRequests);
					commandBuffer->pipelineBarrier(sugg.srcStageMask, sugg.dstStageMask, asset::EDF_NONE, 0u, nullptr, barrierCount, barriers, 0u, nullptr);
				}
				transformTreeManager->updateLocalAndRecomputeGlobalTransforms(baseParams,update;
				*/
				scene::ITransformTreeManager::DispatchParams recomputeDispatch;
				recomputeDispatch.indirect.buffer = nullptr;
				recomputeDispatch.direct.nodeCount = totalJointCount;
				transformTreeManager->recomputeGlobalTransforms(baseParams,recomputeDispatch,ttmDescriptorSets.recomputeGlobal.get());
				// barrier between TTM recompute and TTM recompute+update 
				{
					auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_POST_GLOBAL_TFORM_RECOMPUTE);
					sugg.dstStageMask |= renderingStages; // also also TTM recompute and rendering shader (to read the global transforms)
					uint32_t barrierCount = 0u;
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::relative_transform_prop_ix), sugg.relativeTransforms);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::modified_stamp_prop_ix), sugg.modifiedTimestamps);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::global_transform_prop_ix), sugg.globalTransforms);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::recomputed_stamp_prop_ix), sugg.recomputedTimestamps);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTreeWithNormalMatrices::normal_matrix_prop_ix), sugg.normalMatrices);
					commandBuffer->pipelineBarrier(sugg.srcStageMask, sugg.dstStageMask, asset::EDF_NONE, 0u, nullptr, barrierCount, barriers, 0u, nullptr);
				}
			}

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

			{
				auto nodeIDs = allSkeletonNodesBinding;
				nodeIDs.offset = sizeof(uint32_t);

				scene::ITransformTreeManager::DebugPushConstants pc;
				pc.viewProjectionMatrix = viewProjectionMatrix;
				pc.lineColor.set(0.f,1.f,0.f,1.f);
				pc.aabbColor.set(1.f,0.f,0.f,1.f);
				transformTreeManager->debugDraw(commandBuffer.get(),ttDebugDrawPipeline.get(),transformTree.get(),ttmDescriptorSets.debugDraw.get(),nodeIDs,iotaBinding,pc,totalJointCount);
			}


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

		// transform tree
		core::smart_refctd_ptr<scene::ITransformTreeManager> transformTreeManager;
		core::smart_refctd_ptr<IGPUGraphicsPipeline> ttDebugDrawPipeline;
		scene::ITransformTreeManager::DescriptorSets ttmDescriptorSets;
		// skin cache
		core::smart_refctd_ptr<scene::ISkinInstanceCache> skinInstanceCache;
		core::smart_refctd_ptr<scene::ISkinInstanceCacheManager> sicManager;
		// temporary debug
		core::smart_refctd_ptr<scene::ITransformTreeWithNormalMatrices> transformTree;
		uint32_t totalJointCount;
		SBufferBinding<IGPUBuffer> allSkeletonNodesBinding;
		SBufferBinding<IGPUBuffer> iotaBinding;

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