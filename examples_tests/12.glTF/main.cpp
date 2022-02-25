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
using namespace ui;

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
			
			const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
			const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);
			CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL, "glTF", WIN_W, WIN_H, SC_IMG_COUNT, swapchainImageUsage, surfaceFormat, asset::EF_D32_SFLOAT);
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
			cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
			utilities = std::move(initOutput.utilities);

			transferUpQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
			
			transformTreeManager = scene::ITransformTreeManager::create(utilities.get(),transferUpQueue);
			ttDebugDrawPipeline = transformTreeManager->createDebugPipeline<scene::ITransformTreeWithNormalMatrices>(core::smart_refctd_ptr(renderpass));
			ttmDescriptorSets = transformTreeManager->createAllDescriptorSets(logicalDevice.get());

			sicManager = scene::ISkinInstanceCacheManager::create(utilities.get(),transferUpQueue,transformTreeManager.get());
			sicDebugDrawPipeline = sicManager->createDebugPipeline(core::smart_refctd_ptr(renderpass));
			sicDescriptorSets = sicManager->createAllDescriptorSets(logicalDevice.get());

			nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

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
			core::vector<CompressedAABB> aabbPool;

			const float modelSpacing = 5.f;
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
				const core::aabbox3df defaultAABB(-modelSpacing,-modelSpacing,-modelSpacing,modelSpacing,modelSpacing,modelSpacing);
				aabbPool.push_back(defaultAABB); // compute properly from animations later
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
			constexpr uint32_t MaxNodeCount = 4u<<10u; // get ready for many many nodes
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
				logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(),nbl::video::IGPUCommandBuffer::EL_PRIMARY,1u,&xferCmdbuf);
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
			using intra_skin_instance_counts_t = core::vector<uint32_t>;
			core::vector<intra_skin_instance_counts_t> modelInstanceCounts;
			core::vector<scene::ITransformTree::node_t> allNodes;
			union ModelSkinInstanceInstance
			{
				struct
				{
					uint32_t modelID : 4;
					uint32_t skinID : 5;
					uint32_t instanceID : 23;
				};
				uint32_t keyval;
			};
			core::unordered_map<uint32_t,uint32_t> modelSkinInstanceInstance2NodeID;

			// set up the transform tree
			static_assert(sizeof(std::unique_ptr<uint32_t[]>)==sizeof(void*));
			using node_array_t = const scene::ITransformTree::node_t*;
			core::unordered_map<const asset::ICPUSkeleton*,std::unique_ptr<node_array_t[]>> skeletonInstanceNodes;
			{
				core::unordered_map<const asset::ICPUSkeleton*,uint32_t> skeletonCounters;
				for (const auto& model : models)
				{
					const auto skeletonInstanceCount = std::uniform_int_distribution<uint32_t>(1,5)(mt);
					intra_skin_instance_counts_t instanceCounts(skeletonInstanceCount);
					for (auto& instanceCount : instanceCounts)
						instanceCount = std::uniform_int_distribution<uint32_t>(1,16)(mt);
					modelInstanceCounts.emplace_back() = std::move(instanceCounts);
					for (const auto& skeleton : model.meta->skeletons)
					{
						auto found = skeletonCounters.find(skeleton.get());
						if (found!=skeletonCounters.end())
							found->second += skeletonInstanceCount;
						else
							skeletonCounters.insert({skeleton.get(),skeletonInstanceCount});
					}
				}
				uint32_t waitSemaphoreCount = 0u;
				video::IGPUSemaphore* const* waitSempahores = nullptr;
				const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
				// allocate skeleton instance nodes in TT
				{
					const auto uniqueSkeletonCount = skeletonCounters.size();
					core::vector<const ICPUSkeleton*> skeletons; skeletons.reserve(uniqueSkeletonCount);
					core::vector<uint32_t> skeletonInstanceCounts; skeletonInstanceCounts.reserve(uniqueSkeletonCount);
					for (const auto& pair : skeletonCounters)
					{
						const auto skeleton = pair.first;
						const auto instanceCount = pair.second;
						skeletons.push_back(skeleton);
						skeletonInstanceCounts.push_back(instanceCount);
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
					allNodes.resize(stagingReqs.nodeCount,scene::ITransformTree::invalid_node);
					core::vector<scene::ITransformTree::node_t> parentScratch(stagingReqs.nodeCount);
					core::vector<scene::ITransformTree::relative_transform_t> transformScratch(stagingReqs.transformScratchCount);
					{
						const scene::ITransformTree::node_t* outNodeIt = nullptr;
						for (auto i=0u; i<uniqueSkeletonCount; i++)
						{
							const auto skeleton = skeletons[i];
							const auto instanceCount = skeletonInstanceCounts[i];
							std::unique_ptr<node_array_t[]> instanceNodeLists(new node_array_t[instanceCount]);
							for (auto i=0u; i<instanceCount; i++)
							{
								instanceNodeLists[i] = outNodeIt;
								outNodeIt += skeleton->getJointCount();
							}
							skeletonInstanceNodes[skeleton] = std::move(instanceNodeLists);
						}
					}

					skeletonAllocationRequest.outNodes = allNodes.data();
					skeletonAllocationRequest.parentScratch = parentScratch.data();
					skeletonAllocationRequest.transformScratch = transformScratch.data();

					transformTreeManager->addSkeletonNodes(skeletonAllocationRequest,waitSemaphoreCount,waitSempahores,waitStages);
				}
				// allocate pivot nodes (instance nodes)
				{
					pivotNodesRange.offset = allNodes.size();

					core::vector<core::matrix3x4SIMD> relativeTransforms;
					for (auto k=0u; k<modelInstanceCounts.size(); k++)
					for (auto j=0u; j<modelInstanceCounts[k].size(); j++)
					for (auto i=0u; i<modelInstanceCounts[k][j]; i++)
					{
						ModelSkinInstanceInstance key;
						key.modelID = k;
						key.skinID = j;
						key.instanceID = i;
						modelSkinInstanceInstance2NodeID[key.keyval] = allNodes.size();
						allNodes.push_back(scene::ITransformTree::invalid_node);
						relativeTransforms.emplace_back().setTranslation(core::vectorSIMDf(i,j,k)*2.1f*modelSpacing);
					}
					pivotNodesRange.size = allNodes.size()-pivotNodesRange.offset;

					scene::ITransformTreeManager::AdditionRequest request;
					request.tree = transformTree.get();
					request.parents = {}; // no parents
					request.relativeTransforms.data = relativeTransforms.data();
					request.relativeTransforms.device2device = false;
					request.cmdbuf = xferCmdbuf.get();
					request.fence = xferFence.get();
					request.scratch = xferScratch;
					request.upBuff = utilities->getDefaultUpStreamingBuffer();
					request.poolHandler = ppHandler;
					request.queue = xferQueue;
					request.logger = logger.get();
					request.outNodes = {allNodes.data()+pivotNodesRange.offset,allNodes.data()+allNodes.size()};
					transformTreeManager->addNodes(request,waitSemaphoreCount,waitSempahores,waitStages);
					
					pivotNodesRange.offset *= sizeof(scene::ITransformTree::node_t);
					pivotNodesRange.size *= sizeof(scene::ITransformTree::node_t);
				}

				// first uint needs to be the count
				allNodes.insert(allNodes.begin(),allNodes.size());
				pivotNodesRange.offset += sizeof(uint32_t);

				auto allNodesBuffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(scene::ITransformTree::node_t)*allNodes.size(),allNodes.data());
				transformTreeManager->updateRecomputeGlobalTransformsDescriptorSet(logicalDevice.get(),ttmDescriptorSets.recomputeGlobal.get(),{0ull,allNodesBuffer});
				pivotNodesRange.buffer = std::move(allNodesBuffer);
			}
			struct InverseBindPoseRangeHash
			{
				inline size_t operator()(const asset::SBufferRange<const asset::ICPUBuffer>& inverseBindPoseRange) const
				{
					return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&inverseBindPoseRange),sizeof(inverseBindPoseRange)));
				}
			};
			core::unordered_map<asset::SBufferRange<const asset::ICPUBuffer>,std::unique_ptr<uint32_t[]>,InverseBindPoseRangeHash> inverseBindPoseRanges;
			struct Skin
			{
				const ICPUSkeleton* skeleton;
				asset::SBufferBinding<const asset::ICPUBuffer> skinTranslationTable;
				asset::SBufferBinding<const asset::ICPUBuffer> inverseBindPoses;
				uint32_t jointCount;

				inline bool operator==(const Skin& other) const
				{
					return skeleton==other.skeleton && skinTranslationTable==other.skinTranslationTable && inverseBindPoses==other.inverseBindPoses && jointCount==other.jointCount;
				}
			};
			struct SkinHash
			{
				inline size_t operator()(const Skin& skin) const
				{
					return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&skin),sizeof(skin)));
				}
			};
			core::unordered_map<Skin,uint32_t,SkinHash> skinCounters;
			struct AABBRangeHash
			{
				inline size_t operator()(const asset::SBufferRange<const asset::ICPUBuffer>& aabbRange) const
				{
					return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&aabbRange),sizeof(aabbRange)));
				}
			};
			core::unordered_map<asset::SBufferRange<const asset::ICPUBuffer>,uint32_t,AABBRangeHash> aabbRanges;
			// pick a scene and flag all skin instances
			for (auto i=0u; i<models.size(); i++)
			{
				const uint32_t skeletonInstanceCount = modelInstanceCounts[i].size();
				const auto* meta = models[i].meta;
				// pick a scene
				const auto& scenes = meta->scenes;
				const auto sceneID = meta->defaultSceneID<scenes.size() ? meta->defaultSceneID:0u;
				const auto& scene = scenes[sceneID];
				for (const auto& instanceID : scene.instanceIDs)
				{
					const auto& instance = meta->instances[instanceID];
					for (const auto& meshbuffer : instance.mesh->getMeshBuffers())
					{
						const uint32_t jointCount = meshbuffer->getJointCount();
						if (jointCount==0u)
							continue;
						
						asset::SBufferRange<const asset::ICPUBuffer> aabbRange;
						aabbRange.offset = meshbuffer->getJointAABBBufferBinding().offset;
						aabbRange.size = sizeof(core::aabbox3df)*jointCount;
						aabbRange.buffer = meshbuffer->getJointAABBBufferBinding().buffer;
						auto foundAABBR = aabbRanges.find(aabbRange);
						if (aabbRange.buffer)
						{
							if (foundAABBR==aabbRanges.end())
							{
								auto aabbs = reinterpret_cast<const core::aabbox3df*>(reinterpret_cast<const uint8_t*>(aabbRange.buffer->getPointer())+aabbRange.offset);
								foundAABBR = aabbRanges.insert({std::move(aabbRange),aabbPool.size()}).first;
								for (auto j=0u; j<jointCount; j++)
									aabbPool.emplace_back(aabbs[j]);
							}
						}

						asset::SBufferRange<const asset::ICPUBuffer> inverseBindPoseRange;
						inverseBindPoseRange.offset = meshbuffer->getInverseBindPoseBufferBinding().offset;
						inverseBindPoseRange.size = sizeof(scene::ISkinInstanceCache::inverse_bind_pose_t)*jointCount;
						inverseBindPoseRange.buffer = meshbuffer->getInverseBindPoseBufferBinding().buffer;
						if (inverseBindPoseRange.buffer)
						{
							auto foundIBPR = inverseBindPoseRanges.find(inverseBindPoseRange);
							if (foundIBPR==inverseBindPoseRanges.end())
								inverseBindPoseRanges.insert({std::move(inverseBindPoseRange),std::unique_ptr<uint32_t[]>(new uint32_t[jointCount])});
						}

						Skin skin = {instance.skeleton,instance.skinTranslationTable,meshbuffer->getInverseBindPoseBufferBinding(),jointCount};
						auto foundSkin = skinCounters.find(skin);
						if (foundSkin!= skinCounters.end())
							foundSkin->second += skeletonInstanceCount;
						else
							skinCounters.insert({std::move(skin),skeletonInstanceCount});
					}
				}
			}
			// transfer compressed aabbs to the GPU
			{
				aabbBinding = {0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(CompressedAABB)*aabbPool.size(),aabbPool.data())};
				transformTreeManager->updateDebugDrawDescriptorSet(logicalDevice.get(),ttmDescriptorSets.debugDraw.get(),SBufferBinding(aabbBinding));
				
				IGPUBuffer::SCreationParams params = {};
				params.usage = core::bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT)|IGPUBuffer::EUF_VERTEX_BUFFER_BIT;
				iotaBinding = {0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,pivotNodesRange.size/sizeof(scene::ITransformTree::node_t))};
			}
			// allocate an inverse bind pose for every inverseBindPose
			{
				auto* ibpPool = skinInstanceCache->getInverseBindPosePool();
				for (const auto& pair : inverseBindPoseRanges)
				{
					const auto& ibpr = pair.first;
					const uint8_t* ptr = reinterpret_cast<const uint8_t*>(ibpr.buffer->getPointer())+ibpr.offset;
					auto inverseBindPoseIt = reinterpret_cast<const scene::ISkinInstanceCache::inverse_bind_pose_t*>(ptr);
					auto end = reinterpret_cast<const scene::ISkinInstanceCache::inverse_bind_pose_t*>(ptr+ibpr.size);
					const auto jointCount = std::distance(inverseBindPoseIt,end);

					//
					uint32_t* ids = pair.second.get();
					std::fill_n(ids,jointCount,video::IPropertyPool::invalid);
					ibpPool->allocateProperties(ids,ids+jointCount);

					video::CPropertyPoolHandler::UpStreamingRequest request = {};
					request.setFromPool(ibpPool,scene::ISkinInstanceCache::inverse_bind_pose_prop_ix);
					request.fill = false;
					request.elementCount = jointCount;
					request.source.data = inverseBindPoseIt;
					request.source.device2device = false;
					request.dstAddresses = ids;
					request.srcAddresses = nullptr; //iota
					auto* pRequest = &request;
					uint32_t waitSemaphoreCount = 0u;
					video::IGPUSemaphore* const* waitSempahores = nullptr;
					const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
					ppHandler->transferProperties(
						utilities->getDefaultUpStreamingBuffer(),xferCmdbuf.get(),xferFence.get(),xferQueue,xferScratch,
						pRequest,1u,waitSemaphoreCount,waitSempahores,waitStages,logger.get()
					);
				}
			}
			core::vector<std::unique_ptr<scene::ISkinInstanceCache::skin_instance_t[]>> skinInstances;
			{
				// allocate a skin cache entry for every skin instance
				{
					const auto skinCount = skinCounters.size();
					skinInstances.reserve(skinCount);
					core::vector<uint32_t> skinJointCounts; skinJointCounts.reserve(skinCount);
					core::vector<uint32_t> instanceCounts; instanceCounts.reserve(skinCount);
					core::vector<const uint32_t*> translationTables; translationTables.reserve(skinCount);
					core::vector<const scene::ITransformTree::node_t* const*> skeletonNodes;
					core::vector<const scene::ISkinInstanceCache::inverse_bind_pose_offset_t*> inverseBindPoseRangesFlatArray; inverseBindPoseRangesFlatArray.reserve(skinCount);
					for (auto& pair : skinCounters)
					{
						const Skin& skin = pair.first;
						const auto instanceCount = pair.second;

						std::fill_n(skinInstances.emplace_back(new scene::ISkinInstanceCache::skin_instance_t[instanceCount]).get(),instanceCount,scene::ISkinInstanceCache::invalid_instance);
						skinJointCounts.push_back(skin.jointCount);
						instanceCounts.push_back(instanceCount);
						
						const auto* buffPtr = reinterpret_cast<const uint8_t*>(skin.skinTranslationTable.buffer->getPointer());
						translationTables.push_back(reinterpret_cast<const uint32_t*>(buffPtr+skin.skinTranslationTable.offset));
						
						// little remapping
						for (auto& pair2 : skeletonInstanceNodes)
						for (auto i=0u; i<instanceCount; i++)
						{
							auto& arr = pair2.second.get()[i];
							arr = allNodes.data()+1u+std::distance<node_array_t>(nullptr,arr);
						}
						skeletonNodes.push_back(reinterpret_cast<const scene::ITransformTree::node_t* const*>(skeletonInstanceNodes.find(skin.skeleton)->second.get()));
						
						asset::SBufferRange<const asset::ICPUBuffer> ibprKey;
						ibprKey.offset = skin.inverseBindPoses.offset;
						ibprKey.size = sizeof(scene::ISkinInstanceCache::inverse_bind_pose_t)*skin.jointCount;
						ibprKey.buffer = skin.inverseBindPoses.buffer;
						inverseBindPoseRangesFlatArray.push_back(inverseBindPoseRanges.find(ibprKey)->second.get());
					}
					auto pSkinInstances = reinterpret_cast<scene::ISkinInstanceCache::skin_instance_t* const*>(skinInstances.data());

					scene::ISkinInstanceCacheManager::AdditionRequest request;
					request.cache = skinInstanceCache.get();
					request.cmdbuf = xferCmdbuf.get();
					request.fence = xferFence.get();
					request.scratch = xferScratch;
					request.upBuff = utilities->getDefaultUpStreamingBuffer();
					request.poolHandler = ppHandler;
					request.queue = xferQueue;
					request.allocation.skinInstances = {pSkinInstances,pSkinInstances+skinInstances.size()};
					request.allocation.jointCountPerSkin = skinJointCounts.data();
					request.allocation.instanceCounts = instanceCounts.data();
					request.translationTables = translationTables.data();
					request.skeletonNodes = skeletonNodes.data();
					request.inverseBindPoseOffsets = inverseBindPoseRangesFlatArray.data();

					totalJointCount = request.computeStagingRequirements();
					auto jointIndexScratch = std::unique_ptr<scene::ISkinInstanceCache::joint_t[]>(new scene::ISkinInstanceCache::joint_t[totalJointCount]);
					auto jointNodeScratch = std::unique_ptr<scene::ITransformTree::node_t[]>(new scene::ITransformTree::node_t[totalJointCount]);
					auto inverseBindPoseOffsetScratch = std::unique_ptr<scene::ISkinInstanceCache::inverse_bind_pose_offset_t[]>(new scene::ISkinInstanceCache::inverse_bind_pose_offset_t[totalJointCount]);
					request.jointIndexScratch = jointIndexScratch.get();
					request.jointNodeScratch = jointNodeScratch.get();
					request.inverseBindPoseOffsetScratch = inverseBindPoseOffsetScratch.get();
					request.logger = logger.get();

					// set up the transfers of property data for skin instances
					uint32_t waitSemaphoreCount = 0u;
					video::IGPUSemaphore* const* waitSempahores = nullptr;
					const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
					sicManager->addSkinInstances(request,waitSemaphoreCount,waitSempahores,waitStages);

					core::vector<scene::ISkinInstanceCache::skin_instance_t> skinsToUpdate;
					core::vector<uint32_t> jointCountInclusivePrefixSum;
					for (auto i=0u; i<skinCount; i++)
					{
						const auto jointCount = skinJointCounts[i];
						const auto instanceCount = instanceCounts[i];
						for (auto j=0u; j<instanceCount; j++)
						{
							const auto skinInstance = skinInstances[i][j];
							skinsToUpdate.push_back(skinInstance);
						}
						jointCountInclusivePrefixSum.insert(jointCountInclusivePrefixSum.end(),instanceCount,jointCount);
					}
					// first uint needs to be the count
					skinsToUpdate.insert(skinsToUpdate.begin(),skinsToUpdate.size());
					std::inclusive_scan(jointCountInclusivePrefixSum.begin(),jointCountInclusivePrefixSum.end(),jointCountInclusivePrefixSum.begin());
					sicManager->updateCacheUpdateDescriptorSet(
						logicalDevice.get(),sicDescriptorSets.cacheUpdate.get(),
						{0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(scene::ISkinInstanceCache::skin_instance_t)*skinsToUpdate.size(),skinsToUpdate.data())},
						{0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(uint32_t)*jointCountInclusivePrefixSum.size(),jointCountInclusivePrefixSum.data())}
					);
				}
				// debug draw skin instances
				{
					core::vector<scene::ISkinInstanceCacheManager::DebugDrawData> debugData;
					core::vector<uint32_t> jointCountInclPrefixSum;
					for (auto i=0u; i<models.size(); i++)
					{
						const uint32_t skeletonInstanceCount = modelInstanceCounts[i].size();
						const auto* meta = models[i].meta;
						// pick a scene
						const auto& scenes = meta->scenes;
						const auto sceneID = meta->defaultSceneID<scenes.size() ? meta->defaultSceneID:0u;
						const auto& scene = scenes[sceneID];
						for (const auto& instanceID : scene.instanceIDs)
						{
							const auto& instance = meta->instances[instanceID];
							for (const auto& meshbuffer : instance.mesh->getMeshBuffers())
							{
								const uint32_t jointCount = meshbuffer->getJointCount();
								if (jointCount==0u)
									continue;
						
								for (auto j=0u; j<skeletonInstanceCount; j++)
								for (auto k=0u; k<modelInstanceCounts[i][j]; k++)
								{
									ModelSkinInstanceInstance key;
									key.modelID = i;
									key.skinID = j;
									key.instanceID = k;
									
									auto& debugDrawData = debugData.emplace_back();
									debugDrawData.skinOffset = 0u; // TODO
									debugDrawData.aabbOffset = 1u; // TODO
									debugDrawData.pivotNode = modelSkinInstanceInstance2NodeID[key.keyval];
									jointCountInclPrefixSum.push_back(jointCount);
								}
							}
						}
					}
					totalDrawInstances = debugData.size();
					std::inclusive_scan(jointCountInclPrefixSum.begin(),jointCountInclPrefixSum.end(),jointCountInclPrefixSum.begin());
					totalJointInstances = jointCountInclPrefixSum.back();

					sicManager->updateDebugDrawDescriptorSet(
						logicalDevice.get(),sicDescriptorSets.debugDraw.get(),
						transformTree.get(),SBufferBinding(aabbBinding),
						{0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(scene::ISkinInstanceCacheManager::DebugDrawData)*debugData.size(),debugData.data())},
						{0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,sizeof(uint32_t)*jointCountInclPrefixSum.size(),jointCountInclPrefixSum.data())}
					);
				}
			}

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

			// TODO: Animation playback
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
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.01f, 10000.0f);
			camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 0.4f, 1.f);
			auto lastTime = std::chrono::system_clock::now();

			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,commandBuffers);

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

			constexpr auto MaxBarrierCount = 6u;
			static_assert(MaxBarrierCount>=scene::ITransformTreeManager::SBarrierSuggestion::MaxBufferCount);
			static_assert(MaxBarrierCount>=scene::ISkinInstanceCacheManager::SBarrierSuggestion::MaxBufferCount);
			video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[MaxBarrierCount];
			auto setBufferBarrier = [&barriers,commandBuffer](const uint32_t ix, const asset::SBufferRange<video::IGPUBuffer>& range, const asset::SMemoryBarrier& barrier)
			{
				barriers[ix].barrier = barrier;
				barriers[ix].dstQueueFamilyIndex = barriers[ix].srcQueueFamilyIndex = commandBuffer->getQueueFamilyIndex();
				barriers[ix].buffer = range.buffer;
				barriers[ix].offset = range.offset;
				barriers[ix].size = range.size;
			};
			const core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> renderingStages = asset::EPSF_VERTEX_SHADER_BIT;
			// Update node transforms 
			{
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
				recomputeDispatch.direct.nodeCount = allNodesBinding.size/sizeof(scene::ITransformTree::node_t);
				transformTreeManager->recomputeGlobalTransforms(baseParams,recomputeDispatch,ttmDescriptorSets.recomputeGlobal.get());
				// barrier between TTM recompute and SkinCache Update, TTM recompute+update 
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
			// Update Skin Cache
			{
				scene::ISkinInstanceCacheManager::CacheUpdateParams params;
				params.cmdbuf = commandBuffer.get();
				params.cache = skinInstanceCache.get();
				params.cacheUpdateDS = sicDescriptorSets.cacheUpdate.get();
				params.dispatchDirect.totalJointCount = totalJointCount;
				params.logger = initOutput.logger.get();
				sicManager->cacheUpdate(params);
				// barrier between TTM recompute and TTM recompute+update 
				{
					auto sugg = scene::ISkinInstanceCacheManager::barrierHelper(scene::ISkinInstanceCacheManager::SBarrierSuggestion::EF_POST_CACHE_UPDATE);
					sugg.dstStageMask |= renderingStages;
					uint32_t barrierCount = 0u;
					// only needed if you expect to modify these buffers within the same submit (submit to submit doesnt need any memory barriers)
					//setBufferBarrier(barrierCount++, skinsToUpdateBlock, sugg.skinsToUpdate);
					//setBufferBarrier(barrierCount++, jointCountInclPrefixSum, sugg.jointCountInclPrefixSum);
					setBufferBarrier(barrierCount++, skinInstanceCache->getSkinningMatrixMemoryBlock(), sugg.skinningTransforms);
					setBufferBarrier(barrierCount++, skinInstanceCache->getRecomputedTimestampMemoryBlock(), sugg.skinningRecomputedTimestamps);
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
				scene::ITransformTreeManager::DebugPushConstants pc;
				pc.viewProjectionMatrix = viewProjectionMatrix;
				pc.lineColor.set(0.f,1.f,0.f,1.f);
				pc.aabbColor.set(0.f,0.f,1.f,1.f);
				transformTreeManager->debugDraw(
					commandBuffer.get(),ttDebugDrawPipeline.get(),transformTree.get(),ttmDescriptorSets.debugDraw.get(),
					{pivotNodesRange.offset,pivotNodesRange.buffer},iotaBinding,pc,pivotNodesRange.size/sizeof(scene::ITransformTree::node_t)
				);
			}
			{
				scene::ISkinInstanceCacheManager::DebugPushConstants pc;
				pc.viewProjectionMatrix = viewProjectionMatrix;
				pc.lineColor.set(0.f,1.f,0.f,1.f);
				pc.aabbColor.set(1.f,0.f,0.f);
				pc.skinCount = totalDrawInstances;
				sicManager->debugDraw(commandBuffer.get(),sicDebugDrawPipeline.get(),skinInstanceCache.get(),sicDescriptorSets.debugDraw.get(),pc,totalJointInstances);
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
		CommonAPI::InitOutput initOutput;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
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
		core::smart_refctd_ptr<scene::ITransformTreeWithNormalMatrices> transformTree;
		SBufferRange<IGPUBuffer> allNodesBinding;
		// transform tree debug draw
		SBufferRange<IGPUBuffer> pivotNodesRange;
		SBufferBinding<IGPUBuffer> iotaBinding;
		// neither skin not transform tree
		SBufferBinding<IGPUBuffer> aabbBinding;
		// skin cache
		core::smart_refctd_ptr<scene::ISkinInstanceCacheManager> sicManager;
		core::smart_refctd_ptr<IGPUGraphicsPipeline> sicDebugDrawPipeline;
		scene::ISkinInstanceCacheManager::DescriptorSets sicDescriptorSets;
		core::smart_refctd_ptr<scene::ISkinInstanceCache> skinInstanceCache;
		uint32_t totalJointCount;
		// skin cache debug draw
		uint32_t totalDrawInstances;
		uint32_t totalJointInstances;

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