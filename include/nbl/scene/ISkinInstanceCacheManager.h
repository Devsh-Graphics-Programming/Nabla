// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SCENE_I_SKIN_INSTANCE_CACHE_MANAGER_H_INCLUDED_
#define _NBL_SCENE_I_SKIN_INSTANCE_CACHE_MANAGER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/video/declarations.h"

#include "nbl/core/definitions.h"

#include "nbl/scene/ISkinInstanceCache.h"

namespace nbl::scene
{


class ISkinInstanceCacheManager : public virtual core::IReferenceCounted
{
	public:
		// creation
        static inline core::smart_refctd_ptr<ISkinInstanceCacheManager> create(video::IUtilities* utils, video::IQueue* uploadQueue, ITransformTreeManager* ttm)
        {
			auto device = utils->getLogicalDevice();
			auto system = device->getPhysicalDevice()->getSystem();
			auto createShader = [&system,&device](auto uniqueString, asset::IShader::E_SHADER_STAGE type=asset::IShader::ESS_COMPUTE) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
			{
				auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
				{
					nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
					system->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
					if (future.wait())
						return future.copy();
					return nullptr;
				};

				auto glslFile = loadBuiltinData(uniqueString);
				core::smart_refctd_ptr<asset::ICPUBuffer> glsl;
				{
					glsl = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
					memcpy(glsl->getPointer(), glslFile->getMappedPointer(), glsl->getSize());
				}
				auto shader = device->createShader(core::make_smart_refctd_ptr<asset::ICPUShader>(core::smart_refctd_ptr(glsl), type, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, ""));
				return device->createSpecializedShader(shader.get(),{nullptr,nullptr,"main"});
			};

			auto updateSpec = createShader("nbl/builtin/glsl/skinning/cache_update.comp");
			auto debugDrawVertexSpec = createShader("nbl/builtin/glsl/skinning/debug.vert",asset::IShader::ESS_VERTEX);
			auto debugDrawFragmentSpec = createShader("nbl/builtin/material/debug/vertex_normal/specialized_shader.frag",asset::IShader::ESS_FRAGMENT);
			if (!updateSpec || !debugDrawVertexSpec || !debugDrawFragmentSpec)
				return nullptr;

			const auto& limits = device->getPhysicalDevice()->getLimits();
			core::vector<uint8_t> tmp(limits.minSSBOAlignment);
			*reinterpret_cast<ISkinInstanceCache::recomputed_stamp_t*>(tmp.data()) = ISkinInstanceCache::initial_recomputed_timestamp;
			
			video::IGPUBuffer::SCreationParams initTimestampValueBufferCreationParams = {};
			initTimestampValueBufferCreationParams.size = tmp.size();
			initTimestampValueBufferCreationParams.usage = core::bitflag<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT);
			auto initTimestampValue = utils->createFilledDeviceLocalBufferOnDedMem(uploadQueue,std::move(initTimestampValueBufferCreationParams),tmp.data());
			if (!initTimestampValue)
				return nullptr;
			initTimestampValue->setObjectDebugName("ISkinInstanceCacheManager::m_initTimestampValue");
			
			auto cacheDsLayout = ISkinInstanceCache::createCacheDescriptorSetLayout(device);
			auto cacheUpdateDsLayout = createCacheUpdateDescriptorSetLayout(device);
			auto debugDrawDsLayout = createDebugDrawDescriptorSetLayout(device);
			if (!cacheDsLayout || !cacheUpdateDsLayout || !debugDrawDsLayout)
				return nullptr;
			
			auto cacheUpdateLayout = device->createPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(cacheDsLayout),std::move(cacheUpdateDsLayout));
			asset::SPushConstantRange pcRange;
			pcRange.offset = 0u;
			pcRange.size = sizeof(DebugPushConstants);
			pcRange.stageFlags = asset::IShader::ESS_VERTEX;
			auto debugDrawLayout = device->createPipelineLayout(&pcRange,&pcRange+1u,core::smart_refctd_ptr(cacheDsLayout),std::move(debugDrawDsLayout));

			auto cacheUpdatePipeline = device->createComputePipeline(nullptr,std::move(cacheUpdateLayout),std::move(updateSpec));
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> debugDrawIndepenedentPipeline;
			{
				asset::SVertexInputParams vertexInputParams = {};
				asset::SBlendParams blendParams = {};
				asset::SPrimitiveAssemblyParams primitiveAssemblyParams = {};
				primitiveAssemblyParams.primitiveType = asset::EPT_LINE_LIST;
				asset::SRasterizationParams rasterizationParams = {};
				rasterizationParams.depthTestEnable = false;

				video::IGPUSpecializedShader* const debugDrawShaders[] = {debugDrawVertexSpec.get(),debugDrawFragmentSpec.get()};
				debugDrawIndepenedentPipeline = device->createRenderpassIndependentPipeline(
					nullptr,std::move(debugDrawLayout),debugDrawShaders,debugDrawShaders+2u,vertexInputParams,blendParams,primitiveAssemblyParams,rasterizationParams
				);
			}
			if (!cacheUpdatePipeline || !debugDrawIndepenedentPipeline)
				return nullptr;

			auto* sicm = new ISkinInstanceCacheManager(
				core::smart_refctd_ptr<video::ILogicalDevice>(device),ttm,
				std::move(cacheUpdatePipeline),std::move(debugDrawIndepenedentPipeline),std::move(initTimestampValue)
			);
            return core::smart_refctd_ptr<ISkinInstanceCacheManager>(sicm,core::dont_grab);
        }

		static inline constexpr uint32_t TransferCount = 3u;
		struct RequestBase
		{
			ISkinInstanceCache* cache;
		};
		//
		struct TransferRequest : RequestBase
		{
			asset::SBufferRange<video::IGPUBuffer> jointCacheOffsets = {};
			asset::SBufferBinding<video::IGPUBuffer> jointNodes = {};
			asset::SBufferBinding<video::IGPUBuffer> inverseBindPoseOffsets = {};
		};
		inline bool setupTransfers(const TransferRequest& request, video::CPropertyPoolHandler::TransferRequest* transfers)
		{
			if (!request.cache)
				return false;

			for (auto i=0u; i<TransferCount; i++)
			{
				transfers[i].elementCount = request.jointCacheOffsets.size/sizeof(ISkinInstanceCache::joint_t);
				transfers[i].srcAddressesOffset = video::IPropertyPool::invalid; // iota input indices
				transfers[i].dstAddressesOffset = request.jointCacheOffsets.offset;
			}
			transfers[0].memblock = request.cache->getJointNodeMemoryBlock();
			transfers[0].elementSize = sizeof(ISkinInstanceCache::joint_t);
			transfers[0].flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			transfers[0].buffer = request.jointNodes;
			transfers[1].memblock = request.cache->getRecomputedTimestampMemoryBlock();
			transfers[1].elementSize = sizeof(ISkinInstanceCache::recomputed_stamp_t);
			transfers[1].flags = video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[1].buffer = {0ull,m_initialTimestampBuffer};
			transfers[2].memblock = request.cache->getInverseBindPoseOffsetMemoryBlock();
			transfers[2].elementSize = sizeof(ISkinInstanceCache::inverse_bind_pose_offset_t);
			transfers[2].flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			transfers[2].buffer = request.inverseBindPoseOffsets;
			return true;
		}

		//
		struct UpstreamRequest : RequestBase
		{
			core::SRange<const ISkinInstanceCache::skin_instance_t> jointCacheOffsets = { nullptr,nullptr };
			video::CPropertyPoolHandler::UpStreamingRequest::Source jointNodes = {};
			video::CPropertyPoolHandler::UpStreamingRequest::Source inverseBindPoseOffsets = {};
		};
		inline bool setupTransfers(const UpstreamRequest& request, video::CPropertyPoolHandler::UpStreamingRequest* upstreams)
		{
			if (!request.cache)
				return false;

			for (auto i=0u; i<TransferCount; i++)
			{
				upstreams[i].elementCount = request.jointCacheOffsets.size();
				upstreams[i].srcAddresses = nullptr;
				upstreams[i].dstAddresses = request.jointCacheOffsets.begin();
			}
			if (request.jointCacheOffsets.empty())
				return true;

			upstreams[0].destination = request.cache->getJointNodeMemoryBlock();
			upstreams[0].fill = false;
			upstreams[0].elementSize = sizeof(ISkinInstanceCache::joint_t);
			upstreams[0].source = request.jointNodes;
			upstreams[1].destination = request.cache->getRecomputedTimestampMemoryBlock();
			upstreams[1].fill = true;
			upstreams[1].elementSize = sizeof(ISkinInstanceCache::recomputed_stamp_t);
			upstreams[1].source.buffer = {0ull,m_initialTimestampBuffer};
			upstreams[2].destination = request.cache->getInverseBindPoseOffsetMemoryBlock();
			upstreams[2].fill = false;
			upstreams[2].elementSize = sizeof(ISkinInstanceCache::inverse_bind_pose_offset_t);
			upstreams[2].source = request.inverseBindPoseOffsets;
			return true;
		}

		//
		struct AdditionRequest : RequestBase
		{
			inline bool isValid() const
			{
				return cache && cmdbuf && fence && scratch.isValid() && upBuff && poolHandler && queue &&
					allocation.isValid() && translationTables && skeletonNodes && inverseBindPoseOffsets;
			}

			// return `totalJointCount`
			inline uint32_t computeStagingRequirements() const
			{
				uint32_t totalJointCount = 0u;
				for (auto i=0u; i<allocation.skinInstances.size(); i++)
				{
					const auto jointCount = allocation.jointCountPerSkin[i];
					const auto instanceCount = allocation.instanceCounts ? allocation.instanceCounts[i]:1u;
					totalJointCount += jointCount*instanceCount;
				}
				return totalJointCount;
			}

			
			// must be in recording state
			video::IGPUCommandBuffer* cmdbuf;
			video::IGPUFence* fence;
			asset::SBufferBinding<video::IGPUBuffer> scratch;
			video::StreamingTransientDataBufferMT<>* upBuff;
			video::CPropertyPoolHandler* poolHandler;
			video::IQueue* queue;

			ISkinInstanceCache::Allocation allocation;
			// one entry for each joint in the 2D array of `skinInstances` (all instances share)
			const uint32_t* const* translationTables;
			// one array for each skin's instance, order is [skinID][instanceID][localNodeID]
			const ITransformTree::node_t* const* const* skeletonNodes;
			// one inverse bind pose offset array per skin (all instances share)
			const ISkinInstanceCache::inverse_bind_pose_offset_t* const* inverseBindPoseOffsets;
			// scratch buffers are just required to be `totalJointCount` sized, but they can be filled with garbage
			ISkinInstanceCache::joint_t* jointIndexScratch;
			ITransformTree::node_t* jointNodeScratch;
			ISkinInstanceCache::inverse_bind_pose_offset_t* inverseBindPoseOffsetScratch;

			system::logger_opt_ptr logger = nullptr;
		}; 
		inline uint32_t addSkinInstances(
			const AdditionRequest& request, uint32_t& waitSemaphoreCount,
			video::IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite,
			const asset::E_PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore, 
			const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait()
		)
		{
			if (!request.isValid())
				return false;
			
			const auto totalJointCount = request.computeStagingRequirements();
			if (totalJointCount==0u)
				return true;

			if (!request.cache->allocate(request.allocation))
				return false;

			video::CPropertyPoolHandler::UpStreamingRequest upstreams[TransferCount];
			UpstreamRequest req;
			static_cast<RequestBase&>(req) = request;
			{
				auto jointIndexIt = request.jointIndexScratch;
				auto jointNodeIt = request.jointNodeScratch;
				auto inverseBindPoseOffsetIt = request.inverseBindPoseOffsetScratch;
				for (auto i=0u; i<request.allocation.skinInstances.size(); i++)
				{
					const auto jointCount = request.allocation.jointCountPerSkin[i];
					const auto instanceCount = request.allocation.instanceCounts ? request.allocation.instanceCounts[i]:1u;
					for (auto j=0u; j<instanceCount; j++)
					{
						const auto& skinInstanceIndex = request.allocation.skinInstances.begin()[i][j];
						for (auto k=0u; k<jointCount; k++)
						{
							*(jointIndexIt++) = skinInstanceIndex+k;
							*(jointNodeIt++) = request.skeletonNodes[i][j][request.translationTables[i][k]];
							*(inverseBindPoseOffsetIt++) = request.inverseBindPoseOffsets[i][k];
						}
					}
				}
			}
			req.jointCacheOffsets = {request.jointIndexScratch,request.jointIndexScratch+totalJointCount};
			req.jointNodes.device2device = false;
			req.jointNodes.data = request.jointNodeScratch;
			req.inverseBindPoseOffsets.device2device = false;
			req.inverseBindPoseOffsets.data = request.inverseBindPoseOffsetScratch;
			if (!setupTransfers(req,upstreams))
				return false;
			
			auto upstreamsPtr = upstreams;
			return request.poolHandler->transferProperties(
				request.upBuff,request.cmdbuf,request.fence,request.queue,request.scratch,upstreamsPtr,TransferCount,
				waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,request.logger,maxWaitPoint
			);
		}

		//
		struct SBarrierSuggestion
		{
			static inline constexpr uint32_t MaxBufferCount = 4u;

			//
			enum E_FLAG
			{
				EF_PRE_CACHE_UPDATE = 0x1u,
				EF_POST_CACHE_UPDATE = 0x2u
			};

			core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask = static_cast<asset::E_PIPELINE_STAGE_FLAGS>(0u);
			core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask = static_cast<asset::E_PIPELINE_STAGE_FLAGS>(0u);
			asset::SMemoryBarrier skinsToUpdate = {};
			asset::SMemoryBarrier jointCountInclPrefixSum = {};
			asset::SMemoryBarrier skinningTransforms = {};
			asset::SMemoryBarrier skinningRecomputedTimestamps = {};
		};
		//
		static inline SBarrierSuggestion barrierHelper(const SBarrierSuggestion::E_FLAG type)
		{
			SBarrierSuggestion barrier;

			barrier.srcStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			barrier.dstStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			if (type&SBarrierSuggestion::EF_PRE_CACHE_UPDATE)
			{
				barrier.skinsToUpdate.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.jointCountInclPrefixSum.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				// the case of update stepping on its own toes is handled by the POST case
			}
			if (type&SBarrierSuggestion::EF_POST_CACHE_UPDATE)
			{
				// we also need to barrier against any future update to the inputs overstepping our reading
				barrier.skinsToUpdate.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.jointCountInclPrefixSum.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
			}
			const auto rwAccessMask = core::bitflag(asset::EAF_SHADER_READ_BIT)|asset::EAF_SHADER_WRITE_BIT;
			barrier.skinningTransforms.srcAccessMask |= rwAccessMask;
			barrier.skinningTransforms.dstAccessMask |= rwAccessMask;
			barrier.skinningRecomputedTimestamps.srcAccessMask |= rwAccessMask;
			barrier.skinningRecomputedTimestamps.dstAccessMask |= rwAccessMask;

			return barrier;
		}


		//
		struct CacheUpdateParams
		{
			CacheUpdateParams()
			{
				dispatchIndirect.buffer = nullptr;
				dispatchDirect.totalJointCount = 0u;
			}

			inline CacheUpdateParams& operator=(const CacheUpdateParams& other)
			{
				cmdbuf = other.cmdbuf;
				cache = other.cache;
				if (other.dispatchIndirect.buffer)
					dispatchIndirect = other.dispatchIndirect;
				else
				{
					dispatchIndirect.buffer = nullptr;
					dispatchDirect.totalJointCount = other.dispatchDirect.totalJointCount;
				}
				logger = other.logger;
				return *this;
			}

			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			ISkinInstanceCache* cache;
			const video::IGPUDescriptorSet* cacheUpdateDS;
			union
			{
				struct
				{
					video::IGPUBuffer* buffer;
					uint64_t offset;
				} dispatchIndirect;
				struct
				{
					private:
						uint64_t dummy;
					public:
						uint32_t totalJointCount;
				} dispatchDirect;
			};
			system::logger_opt_ptr logger = nullptr;
		};
		//
		static inline constexpr uint32_t CacheUpdateDescriptorBindingCount = 2u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createCacheUpdateDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[CacheUpdateDescriptorBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,CacheUpdateDescriptorBindingCount,asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createDescriptorSetLayout(bindings,bindings+CacheUpdateDescriptorBindingCount);
		}
		// first uint in the `skinsToUpdate` buffer tells us how many skinCache entries to update we have
		// rest is filled wtih `ISkinInstanceCache::skin_instance_t` (offset to the cont array allocated to store the cache for a skin)
		// `jointCountInclPrefixSum` contains the inclusive prefix sum of each skin's joint count
		static inline void updateCacheUpdateDescriptorSet(
			video::ILogicalDevice* device, video::IGPUDescriptorSet* cacheUpdateDS,
			asset::SBufferBinding<video::IGPUBuffer>&& skinsToUpdate,
			asset::SBufferBinding<video::IGPUBuffer>&& jointCountInclPrefixSum
		)
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet writes[CacheUpdateDescriptorBindingCount];
			video::IGPUDescriptorSet::SDescriptorInfo infos[CacheUpdateDescriptorBindingCount];
			for (auto i=0u; i<CacheUpdateDescriptorBindingCount; i++)
			{
				writes[i].dstSet = cacheUpdateDS;
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				writes[i].info = infos+i;
			}
			infos[0] = skinsToUpdate;
			infos[1] = jointCountInclPrefixSum;
			device->updateDescriptorSets(CacheUpdateDescriptorBindingCount,writes,0u,nullptr);
		}
		inline bool cacheUpdate(const CacheUpdateParams& params)
		{
			const video::IGPUDescriptorSet* descSets[] = { params.cache->getCacheDescriptorSet(),params.cacheUpdateDS };
			params.cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_cacheUpdate->getLayout(),0u,2u,descSets);
			
			params.cmdbuf->bindComputePipeline(m_cacheUpdate.get());
			if (params.dispatchIndirect.buffer)
				params.cmdbuf->dispatchIndirect(params.dispatchIndirect.buffer,params.dispatchIndirect.offset);
			else
			{
				const auto& limits = m_device->getPhysicalDevice()->getLimits();
				params.cmdbuf->dispatch(limits.computeOptimalPersistentWorkgroupDispatchSize(params.dispatchDirect.totalJointCount,m_workgroupSize),1u,1u);
			}
			return true;
		}

		//
		inline auto createDebugPipeline(core::smart_refctd_ptr<const video::IGPURenderpass>&& renderpass)
		{
			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = m_debugDrawRenderpassIndependent;
			graphicsPipelineParams.renderpass = std::move(renderpass);
			return m_device->createGraphicsPipeline(nullptr,std::move(graphicsPipelineParams));
		}

		//
		static inline constexpr uint32_t DebugDrawDescriptorBindingCount = 4u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDebugDrawDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[DebugDrawDescriptorBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,DebugDrawDescriptorBindingCount,asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createDescriptorSetLayout(bindings,bindings+DebugDrawDescriptorBindingCount);
		}
		//
		struct DebugDrawData
		{
			ISkinInstanceCache::skin_instance_t skinOffset;
			uint32_t aabbOffset;
			ITransformTree::node_t pivotNode;
		};
		static inline void updateDebugDrawDescriptorSet(
			video::ILogicalDevice* device, video::IGPUDescriptorSet* debugDrawDS,
			const scene::ITransformTree* transformTree,
			asset::SBufferBinding<video::IGPUBuffer>&& aabbPool,
			asset::SBufferBinding<video::IGPUBuffer>&& skinInstanceDebugData,
			asset::SBufferBinding<video::IGPUBuffer>&& skinInstanceJointCountInclPrefixSum
		)
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet writes[DebugDrawDescriptorBindingCount];
			video::IGPUDescriptorSet::SDescriptorInfo infos[DebugDrawDescriptorBindingCount];
			for (auto i=0u; i<DebugDrawDescriptorBindingCount; i++)
			{
				writes[i].dstSet = debugDrawDS;
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				writes[i].info = infos+i;
			}
			infos[0] = transformTree->getNodePropertyPool()->getPropertyMemoryBlock(scene::ITransformTree::parent_prop_ix);
			infos[1] = aabbPool;
			infos[2] = skinInstanceDebugData;
			infos[3] = skinInstanceJointCountInclPrefixSum;
			device->updateDescriptorSets(DebugDrawDescriptorBindingCount,writes,0u,nullptr);
		}
		struct DebugPushConstants
		{
			core::matrix4SIMD viewProjectionMatrix;
			core::vector4df_SIMD lineColor;
			core::vector3df aabbColor;
			uint32_t skinCount;
		};
		inline void debugDraw(
			video::IGPUCommandBuffer* cmdbuf, const video::IGPUGraphicsPipeline* pipeline, const ISkinInstanceCache* cache,
			const video::IGPUDescriptorSet* debugDrawDS, const DebugPushConstants& pushConstants, const uint32_t totalJointCount
		)
		{
			auto layout = m_debugDrawRenderpassIndependent->getLayout();
			assert(pipeline->getRenderpassIndependentPipeline()->getLayout()==layout);

			const video::IGPUDescriptorSet* sets[] = {cache->getCacheDescriptorSet(),debugDrawDS};
			cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS,layout,0u,2u,sets);
			cmdbuf->bindGraphicsPipeline(pipeline);
			cmdbuf->bindIndexBuffer(m_debugIndexBuffer.get(),0u,asset::EIT_16BIT);
			cmdbuf->pushConstants(layout,asset::IShader::ESS_VERTEX,0u,sizeof(DebugPushConstants),&pushConstants);
			cmdbuf->drawIndexed(ITransformTreeManager::DebugIndexCount,totalJointCount,0u,0u,0u);
		}

		//
		struct DescriptorSets
		{
			core::smart_refctd_ptr<video::IGPUDescriptorSet> cacheUpdate;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> debugDraw;
		};
		DescriptorSets createAllDescriptorSets(video::ILogicalDevice* device)
		{
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> layouts[] =
			{
				createCacheUpdateDescriptorSetLayout(device),
				createDebugDrawDescriptorSetLayout(device)
			};

			auto pool = device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&layouts->get(),&layouts->get()+2u);

			DescriptorSets descSets;
			descSets.cacheUpdate = device->createDescriptorSet(pool.get(),std::move(layouts[0]));
			descSets.debugDraw = device->createDescriptorSet(pool.get(),std::move(layouts[1]));
			return descSets;
		}
	protected:
		ISkinInstanceCacheManager(
			core::smart_refctd_ptr<video::ILogicalDevice>&& _device,
			const ITransformTreeManager* ttm,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& _cacheUpdate,
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>&& _debugDrawRenderpassIndependent,
			core::smart_refctd_ptr<video::IGPUBuffer>&& _initialTimestampBuffer
		) : m_device(std::move(_device)), m_cacheUpdate(std::move(_cacheUpdate)),
			m_debugDrawRenderpassIndependent(std::move(_debugDrawRenderpassIndependent)),
			m_initialTimestampBuffer(std::move(_initialTimestampBuffer)),
			m_debugIndexBuffer(ttm->getDebugIndexBuffer()),
			m_workgroupSize(m_device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations)
		{
		}
		~ISkinInstanceCacheManager()
		{
			// everything drops itself automatically
		}

		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_cacheUpdate;
		core::smart_refctd_ptr<video::IGPUBuffer> m_initialTimestampBuffer;
		const uint32_t m_workgroupSize;

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> m_debugDrawRenderpassIndependent;
		core::smart_refctd_ptr<const video::IGPUBuffer> m_debugIndexBuffer;
};

} // end namespace nbl::scene

#endif

