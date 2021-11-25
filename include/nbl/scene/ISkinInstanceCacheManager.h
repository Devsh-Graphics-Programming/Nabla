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
        static inline core::smart_refctd_ptr<ISkinInstanceCacheManager> create(video::IUtilities* utils, video::IGPUQueue* uploadQueue)
        {
			auto device = utils->getLogicalDevice();
			auto system = device->getPhysicalDevice()->getSystem();
			auto createShader = [&system,&device](auto uniqueString, asset::ISpecializedShader::E_SHADER_STAGE type=asset::ISpecializedShader::ESS_COMPUTE) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
			{
				auto glslFile = system->loadBuiltinData<decltype(uniqueString)>();
				core::smart_refctd_ptr<asset::ICPUBuffer> glsl;
				{
					glsl = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
					memcpy(glsl->getPointer(), glslFile->getMappedPointer(), glsl->getSize());
				}
				auto shader = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(core::smart_refctd_ptr(glsl),asset::IShader::buffer_contains_glsl_t{}));
				return device->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",type });
			};

			auto updateSpec = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/skinning/cache_update.comp")());
			auto debugDrawVertexSpec = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/skinning/debug.vert")(),asset::ISpecializedShader::ESS_VERTEX);
			auto debugDrawFragmentSpec = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/material/debug/vertex_normal/specialized_shader.frag")(),asset::ISpecializedShader::ESS_FRAGMENT);
			if (!updateSpec || !debugDrawVertexSpec || !debugDrawFragmentSpec)
				return nullptr;

			const auto& limits = device->getPhysicalDevice()->getLimits();
			core::vector<uint8_t> tmp(limits.SSBOAlignment);
			*reinterpret_cast<ISkinInstanceCache::recomputed_stamp_t*>(tmp.data()) = ISkinInstanceCache::initial_recomputed_timestamp;
			auto initTimestampValue = utils->createFilledDeviceLocalGPUBufferOnDedMem(uploadQueue,tmp.size(),tmp.data());
			if (!initTimestampValue)
				return nullptr;
			initTimestampValue->setObjectDebugName("ISkinInstanceCacheManager::m_initTimestampValue");

			tmp.resize(sizeof(uint16_t)*IndexCount);
			uint16_t* debugIndices = reinterpret_cast<uint16_t*>(tmp.data());
			{
				std::fill_n(debugIndices,24u,0u);
				debugIndices[0] = 0b000;
				debugIndices[1] = 0b001;
				debugIndices[2] = 0b001;
				debugIndices[3] = 0b011;
				debugIndices[4] = 0b011;
				debugIndices[5] = 0b010;
				debugIndices[6] = 0b010;
				debugIndices[7] = 0b000;
				debugIndices[8] = 0b000;
				debugIndices[9] = 0b100;
				debugIndices[10] = 0b001;
				debugIndices[11] = 0b101;
				debugIndices[12] = 0b010;
				debugIndices[13] = 0b110;
				debugIndices[14] = 0b011;
				debugIndices[15] = 0b111;
				debugIndices[16] = 0b100;
				debugIndices[17] = 0b101;
				debugIndices[18] = 0b101;
				debugIndices[19] = 0b111;
				debugIndices[20] = 0b100;
				debugIndices[21] = 0b110;
				debugIndices[22] = 0b110;
				debugIndices[23] = 0b111;
			}
			auto debugIndexBuffer = utils->createFilledDeviceLocalGPUBufferOnDedMem(uploadQueue,tmp.size(),debugIndices);
			if (!debugIndexBuffer)
				return nullptr;
			
			auto cacheDsLayout = ISkinInstanceCache::createCacheDescriptorSetLayout(device);
			auto cacheUpdateDsLayout = createCacheUpdateDescriptorSetLayout(device);
			auto debugDrawDsLayout = createDebugDrawDescriptorSetLayout(device);
			if (!cacheDsLayout || !cacheUpdateDsLayout || !debugDrawDsLayout)
				return nullptr;
			
			auto cacheUpdateLayout = device->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(cacheDsLayout),std::move(cacheUpdateDsLayout));
			asset::SPushConstantRange pcRange;
			pcRange.offset = 0u;
			pcRange.size = sizeof(DebugPushConstants);
			pcRange.stageFlags = asset::ISpecializedShader::ESS_VERTEX;
			auto debugDrawLayout = device->createGPUPipelineLayout(&pcRange,&pcRange+1u,core::smart_refctd_ptr(cacheDsLayout),std::move(debugDrawDsLayout));

			auto cacheUpdatePipeline = device->createGPUComputePipeline(nullptr,std::move(cacheUpdateLayout),std::move(updateSpec));
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> debugDrawIndepenedentPipeline;
			{
				asset::SVertexInputParams vertexInputParams = {};
				asset::SBlendParams blendParams = {};
				asset::SPrimitiveAssemblyParams primitiveAssemblyParams = {};
				primitiveAssemblyParams.primitiveType = asset::EPT_LINE_LIST;
				asset::SRasterizationParams rasterizationParams = {};
				rasterizationParams.depthTestEnable = false;

				video::IGPUSpecializedShader* const debugDrawShaders[] = {debugDrawVertexSpec.get(),debugDrawFragmentSpec.get()};
				debugDrawIndepenedentPipeline = device->createGPURenderpassIndependentPipeline(
					nullptr,std::move(debugDrawLayout),debugDrawShaders,debugDrawShaders+2u,vertexInputParams,blendParams,primitiveAssemblyParams,rasterizationParams
				);
			}
			if (!cacheUpdatePipeline || !debugDrawIndepenedentPipeline)
				return nullptr;

			auto* sicm = new ISkinInstanceCacheManager(
				core::smart_refctd_ptr<video::ILogicalDevice>(device),
				std::move(cacheUpdatePipeline),std::move(debugDrawIndepenedentPipeline),
				std::move(initTimestampValue),std::move(debugIndexBuffer)
			);
            return core::smart_refctd_ptr<ISkinInstanceCacheManager>(sicm,core::dont_grab);
        }
/*
		static inline constexpr uint32_t TransferCount = 2u;
		struct RequestBase
		{
			ISkinInstanceCache* cache;
		};
		//
		struct TransferRequest : RequestBase
		{
			asset::SBufferBinding<video::IGPUBuffer> joints = {};
		};
		inline bool setupTransfers(const TransferRequest& request, video::CPropertyPoolHandler::TransferRequest* transfers)
		{
			if (!request.tree)
				return false;

			for (auto i=0u; i<TransferCount; i++)
			{
				transfers[i].elementCount = request.joints.size/sizeof(ISkinInstanceCache::joint_t);
				transfers[i].srcAddressesOffset = video::IPropertyPool::invalid;
				transfers[i].dstAddressesOffset = request.joints.offset;
			}
			transfers[0].memblock = request.cache->getJointNodeMemoryBlock();
			transfers[0].elementSize = sizeof(ISkinInstanceCache::joint_t);
			transfers[0].flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			transfers[0].buffer = request.joints;
			transfers[1].memblock = request.cache->getRecomputedTimestampMemoryBlock();
			transfers[1].elementSize = sizeof(ISkinInstanceCache::recomputed_stamp_t);
			transfers[1].flags = video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[1].buffer = m_initialTimestampBufferRange;
			return true;
		}
		
		//
		struct UpstreamRequestBase : RequestBase
		{
			video::CPropertyPoolHandler::UpStreamingRequest::Source joints = {};
		};
		struct UpstreamRequest : UpstreamRequestBase
		{
			core::SRange<const ITransformTree::node_t> nodes = {nullptr,nullptr};
		};
		inline bool setupTransfers(const UpstreamRequest& request, video::CPropertyPoolHandler::UpStreamingRequest* upstreams)
		{
			if (!request.tree)
				return false;
			if (request.nodes.empty())
				return true;

			auto* pool = request.tree->getNodePropertyPool();

			for (auto i=0u; i<TransferCount; i++)
			{
				upstreams[i].elementCount = request.nodes.size();
				upstreams[i].srcAddresses = nullptr;
				upstreams[i].dstAddresses = request.nodes.begin();
			}
			upstreams[0].setFromPool(pool,ITransformTree::parent_prop_ix);
			if (request.parents.device2device || request.parents.data)
			{
				upstreams[0].fill = false;
				upstreams[0].source = request.parents;
			}
			else
			{
				upstreams[0].fill = true;
				upstreams[0].source.buffer = getDefaultValueBufferBinding(ITransformTree::parent_prop_ix);
			}
			upstreams[1].setFromPool(pool,ITransformTree::relative_transform_prop_ix);
			if (request.relativeTransforms.device2device || request.relativeTransforms.data)
			{
				upstreams[1].fill = false;
				upstreams[1].source = request.relativeTransforms;
			}
			else
			{
				upstreams[1].fill = true;
				upstreams[1].source.buffer = getDefaultValueBufferBinding(ITransformTree::relative_transform_prop_ix);
			}
			upstreams[2].setFromPool(pool,ITransformTree::modified_stamp_prop_ix);
			upstreams[2].fill = true;
			upstreams[2].source.buffer = getDefaultValueBufferBinding(ITransformTree::modified_stamp_prop_ix);
			upstreams[3].setFromPool(pool,ITransformTree::recomputed_stamp_prop_ix);
			upstreams[3].fill = true;
			upstreams[3].source.buffer = getDefaultValueBufferBinding(ITransformTree::recomputed_stamp_prop_ix);
			return true;
		}

		struct AdditionRequestBase
		{
			public:
				// must be in recording state
				video::IGPUCommandBuffer* cmdbuf;
				video::IGPUFence* fence;
				asset::SBufferBinding<video::IGPUBuffer> scratch;
				video::StreamingTransientDataBufferMT<>* upBuff;
				video::CPropertyPoolHandler* poolHandler;
				video::IGPUQueue* queue;
				system::logger_opt_ptr logger = nullptr;

			protected:
				inline bool isValid() const
				{
					return cmdbuf && fence && scratch.isValid() && upBuff && poolHandler && queue;
				}
		};
		//
		struct AdditionRequest : UpstreamRequestBase,AdditionRequestBase
		{
			// if the `outSkinInstances` have values not equal to `invalid` then we treat them as already allocated
			// (this allows you to split allocation of nodes from setting up the transfers)
			core::SRange<ISkinInstanceCache::skin_instance_t> outSkinInstances = {nullptr,nullptr};

			inline bool isValid() const
			{
				return AdditionRequestBase::isValid() && outSkinInstances.begin() && outSkinInstances.begin()<= outSkinInstances.end();
			}
		};
		inline uint32_t addNodes(
			const AdditionRequest& request, uint32_t& waitSemaphoreCount,
			video::IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite,
			const asset::E_PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore, 
			const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait()
		)
		{
			if (!request.isValid())
				return false;
			if (request.outSkinInstances.empty())
				return true;

			if (!request.cache->allocate(request.outNodes))
				return false;

			video::CPropertyPoolHandler::UpStreamingRequest upstreams[TransferCount];
			UpstreamRequest req;
			static_cast<UpstreamRequestBase&>(req) = request;
			req.nodes = {request.outNodes.begin(),request.outNodes.end()};
			if (!setupTransfers(req,upstreams))
				return false;

			auto upstreamsPtr = upstreams;
			return request.poolHandler->transferProperties(
				request.upBuff,request.cmdbuf,request.fence,request.queue,request.scratch,upstreamsPtr,TransferCount,
				waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,request.logger,maxWaitPoint
			);
		}

		//
		struct SkeletonAllocationRequest : RequestBase,AdditionRequestBase
		{
			core::SRange<const asset::ICPUSkeleton*> skeletons;
			// if nullptr then treated like a buffer of {1,1,...,1,1}, else needs to be same length as the skeleton range
			const uint32_t* instanceCounts = nullptr;
			// If you make the skeleton hierarchy have a real parent, you won't be able to share it amongst multiple instances of a mesh
			// also in order to render with standard shaders you'll have to cancel out the model transform of the parent for the skinning matrices.
			const ITransformTree::node_t*const * skeletonInstanceParents = nullptr;
			// the following arrays need to be sized according to `StagingRequirements`
			// if the `outNodes` have values not equal to `invalid_node` then we treat them as already allocated
			// (this allows you to split allocation of nodes from setting up the transfers)
			ITransformTree::node_t* outNodes = nullptr;
			// scratch buffers are just required to be the set size, they can be filled with garbage
			ITransformTree::node_t* parentScratch;
			// must be non null if at least one skeleton has default transforms
			ITransformTree::relative_transform_t* transformScratch = nullptr;

			inline bool isValid() const
			{
				return AdditionRequestBase::isValid() && skeletons.begin() && skeletons.begin()<=skeletons.end() && outNodes && parentScratch;
			}

			struct StagingRequirements
			{
				uint32_t nodeCount;
				uint32_t parentScratchSize;
				uint32_t transformScratchSize;
			};
			inline StagingRequirements computeStagingRequirements() const
			{
				StagingRequirements reqs = {0u,0u,0u};
				auto instanceCountIt = instanceCounts;
				for (auto skeleton : skeletons)
				{
					if (skeleton)
					{
						const uint32_t jointCount = skeleton->getJointCount();
						const uint32_t jointInstanceCount = (*instanceCountIt)*jointCount;
						reqs.nodeCount += jointInstanceCount;
						reqs.parentScratchSize += sizeof(ITransformTree::node_t)*jointInstanceCount;
						if (skeleton->getDefaultTransformBinding().buffer)
							reqs.transformScratchSize += sizeof(ITransformTree::relative_transform_t)*jointCount;
					}
					instanceCountIt++;
				}
				if (reqs.transformScratchSize)
					reqs.transformScratchSize += reqs.parentScratchSize*sizeof(uint32_t)/sizeof(ITransformTree::node_t);
				return reqs;
			}
		};
		inline bool addSkeletonNodes(
			const SkeletonAllocationRequest& request, uint32_t& waitSemaphoreCount,
			video::IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite,
			const asset::E_PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore, 
			const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait()
		)
		{
			if (!request.isValid())
				return false;

			const auto staging = request.computeStagingRequirements();
			if (staging.nodeCount==0u)
				return true;

			if (!request.tree->allocateNodes({request.outNodes,request.outNodes+staging.nodeCount}))
				return false;

			uint32_t* const srcTransformIndices = reinterpret_cast<uint32_t*>(request.transformScratch+staging.transformScratchSize/sizeof(ITransformTree::relative_transform_t));
			{
				auto instanceCountIt = request.instanceCounts;
				auto skeletonInstanceParentsIt = request.skeletonInstanceParents;
				auto parentsIt = request.parentScratch;
				auto transformIt = request.transformScratch;
				auto srcTransformIndicesIt = srcTransformIndices;
				uint32_t baseJointInstance = 0u;
				uint32_t baseJoint = 0u;
				for (auto skeleton : request.skeletons)
				{
					const auto instanceCount = request.instanceCounts ? (*(instanceCountIt++)):1u;
					auto* const instanceParents = request.skeletonInstanceParents ? (*(skeletonInstanceParentsIt++)):nullptr;

					const auto jointCount = skeleton->getJointCount();
					auto instanceParentsIt = instanceParents;
					for (auto instanceID=0u; instanceID<instanceCount; instanceID++)
					{
						for (auto jointID=0u; jointID<jointCount; jointID++)
						{
							uint32_t parentID = skeleton->getParentJointID(jointID);
							if (parentID!=asset::ICPUSkeleton::invalid_joint_id)
								parentID = request.outNodes[parentID+baseJointInstance];
							else
								parentID = instanceParents ? (*instanceParentsIt):ITransformTree::invalid_node;
							*(parentsIt++) = parentID;

							if (staging.transformScratchSize)
								*(srcTransformIndicesIt++) = jointID+baseJoint;
						}
						baseJointInstance += jointCount;
					}
					if (skeleton->getDefaultTransformBinding().buffer)
					for (auto jointID=0u; jointID<jointCount; jointID++)
						*(transformIt++) = skeleton->getDefaultTransformMatrix(jointID);
					baseJoint += jointCount;
				}
			}

			video::CPropertyPoolHandler::UpStreamingRequest upstreams[TransferCount];
			UpstreamRequest req = {};
			req.tree = request.tree;
			req.nodes = {request.outNodes,request.outNodes+staging.nodeCount};
			req.parents.data = request.parentScratch;
			if (staging.transformScratchSize)
				req.relativeTransforms.data = request.transformScratch;
			if (!setupTransfers(req,upstreams))
				return false;
			if (staging.transformScratchSize)
				upstreams[1].srcAddresses = srcTransformIndices;

			auto upstreamsPtr = upstreams;
			return request.poolHandler->transferProperties(
				request.upBuff,request.cmdbuf,request.fence,request.queue,request.scratch,upstreamsPtr,TransferCount,
				waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,request.logger,maxWaitPoint
			);
		}


		//
		struct SBarrierSuggestion
		{
			static inline constexpr uint32_t MaxBufferCount = 6u;

			//
			enum E_FLAG
			{
				// basic
				EF_PRE_RELATIVE_TFORM_UPDATE = 0x1u,
				EF_POST_RELATIVE_TFORM_UPDATE = 0x2u,
				EF_PRE_GLOBAL_TFORM_RECOMPUTE = 0x4u,
				EF_POST_GLOBAL_TFORM_RECOMPUTE = 0x8u,
				// if you plan to run recompute right after update
				EF_INBETWEEN_RLEATIVE_UPDATE_AND_GLOBAL_RECOMPUTE = EF_POST_RELATIVE_TFORM_UPDATE|EF_PRE_GLOBAL_TFORM_RECOMPUTE,
				// if you're planning to run the fused recompute and update kernel
				EF_PRE_UPDATE_AND_RECOMPUTE = EF_PRE_RELATIVE_TFORM_UPDATE|EF_PRE_GLOBAL_TFORM_RECOMPUTE,
				EF_POST_UPDATE_AND_RECOMPUTE = EF_POST_RELATIVE_TFORM_UPDATE|EF_POST_GLOBAL_TFORM_RECOMPUTE,
			};

			core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask = static_cast<asset::E_PIPELINE_STAGE_FLAGS>(0u);
			core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask = static_cast<asset::E_PIPELINE_STAGE_FLAGS>(0u);
			asset::SMemoryBarrier requestRanges = {};
			asset::SMemoryBarrier modificationRequests = {};
			asset::SMemoryBarrier relativeTransforms = {};
			asset::SMemoryBarrier modifiedTimestamps = {};
			asset::SMemoryBarrier globalTransforms = {};
			asset::SMemoryBarrier recomputedTimestamps = {};
		};
		//
		static inline SBarrierSuggestion barrierHelper(const SBarrierSuggestion::E_FLAG type)
		{
			const auto rwAccessMask = core::bitflag(asset::EAF_SHADER_READ_BIT)|asset::EAF_SHADER_WRITE_BIT;

			SBarrierSuggestion barrier;
			if (type&SBarrierSuggestion::EF_PRE_RELATIVE_TFORM_UPDATE)
			{
				// we're mostly concerned about stuff writing to buffer update reads from 
				barrier.dstStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				barrier.requestRanges.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modificationRequests.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				// the case of update stepping on its own toes is handled by the POST case
			}
			if (type&SBarrierSuggestion::EF_POST_RELATIVE_TFORM_UPDATE)
			{
				// we're mostly concerned about relative tform update overwriting itself
				barrier.srcStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				barrier.dstStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				// we also need to barrier against any future update to the inputs overstepping our reading
				barrier.requestRanges.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modificationRequests.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				// relative transform can be pre-post multiplied or entirely erased, we're not in charge of that
				// need to also worry about update<->update loop, so both masks are R/W
				barrier.relativeTransforms.srcAccessMask |= rwAccessMask;
				barrier.relativeTransforms.dstAccessMask |= rwAccessMask;
				// we will only overwrite
				barrier.modifiedTimestamps.srcAccessMask |= asset::EAF_SHADER_WRITE_BIT;
				// modified timestamp will be written by previous update, but also has to be read by recompute later
				barrier.modifiedTimestamps.dstAccessMask |= rwAccessMask;
				// we don't touch anything else
			}
			if (type&SBarrierSuggestion::EF_PRE_GLOBAL_TFORM_RECOMPUTE)
			{
				// we're mostly concerned about relative transform update not being finished before global transform recompute runs 
				barrier.srcStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				barrier.dstStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				barrier.relativeTransforms.srcAccessMask |= rwAccessMask;
				barrier.relativeTransforms.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modifiedTimestamps.srcAccessMask |= asset::EAF_SHADER_WRITE_BIT;
				barrier.modifiedTimestamps.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
			}
			if (type&SBarrierSuggestion::EF_POST_GLOBAL_TFORM_RECOMPUTE)
			{
				// we're mostly concerned about global tform recompute overwriting itself
				barrier.srcStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				barrier.dstStageMask |= asset::EPSF_COMPUTE_SHADER_BIT;
				// and future local update overwritng the inputs before recompute is done reading
				barrier.relativeTransforms.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.relativeTransforms.dstAccessMask |= rwAccessMask;
				barrier.modifiedTimestamps.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modifiedTimestamps.dstAccessMask |= asset::EAF_SHADER_WRITE_BIT;
				// global transforms and recompute timestamps can be both read and written
				barrier.globalTransforms.srcAccessMask |= rwAccessMask;
				barrier.globalTransforms.dstAccessMask |= rwAccessMask;
				barrier.recomputedTimestamps.srcAccessMask |= rwAccessMask;
				barrier.recomputedTimestamps.dstAccessMask |= rwAccessMask;
			}
			return barrier;
		}

		//
		using ModificationRequestRange = nbl_glsl_transform_tree_modification_request_range_t;
		struct ParamsBase
		{
			ParamsBase()
			{
				dispatchIndirect.buffer = nullptr;
				dispatchDirect.nodeCount = 0u;
			}
#if 0
			inline ParamsBase& operator=(const ParamsBase& other)
			{
				cmdbuf = other.cmdbuf;
				fence = other.fence;
				tree = other.tree;
				if (other.dispatchIndirect.buffer)
					dispatchIndirect = other.dispatchIndirect;
				else
				{
					dispatchIndirect.buffer = nullptr;
					dispatchDirect.nodeCount = other.dispatchDirect.nodeCount;
				}
				logger = other.logger;
				return *this;
			}
#endif
			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			ISkinInstanceCache* cache;
			// first uint in the buffer tells us how many instances to update we have
			asset::SBufferBinding<video::IGPUBuffer> skinInstanceIDs;
			asset::SBufferBinding<video::IGPUBuffer> boneCountPrefixSum;
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
						uint32_t nodeCount;
				} dispatchDirect;
			};
			system::logger_opt_ptr logger = nullptr;
		};
		inline bool recomputeGlobalTransforms(const GlobalTransformUpdateParams& params)
		{
			const video::IGPUDescriptorSet* descSets[] = { params.tree->getNodePropertyPoolDescriptorSet(),tempDS };
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,pipeline->getLayout(),0u,2u,descSets,nullptr);
			
			cmdbuf->bindComputePipeline(pipeline);
			if (params.dispatchIndirect.buffer)
				cmdbuf->dispatchIndirect(params.dispatchIndirect.buffer,params.dispatchIndirect.offset);
			else
			{
				const auto& limits = m_device->getPhysicalDevice()->getLimits();
				cmdbuf->dispatch(limits.computeOptimalPersistentWorkgroupDispatchSize(params.dispatchDirect.nodeCount,m_workgroupSize),1u,1u);
			}
		}
*/
		//
		inline auto createDebugPipeline(core::smart_refctd_ptr<const video::IGPURenderpass>&& renderpass)
		{
			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = m_debugDrawRenderpassIndependent;
			graphicsPipelineParams.renderpass = std::move(renderpass);
			return m_device->createGPUGraphicsPipeline(nullptr,std::move(graphicsPipelineParams));
		}

		//
		static inline constexpr uint32_t DebugDrawDescriptorBindingCount = 3u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDebugDrawDescriptorSetLayout(video::ILogicalDevice* device, asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[DebugDrawDescriptorBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,DebugDrawDescriptorBindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createGPUDescriptorSetLayout(bindings,bindings+DebugDrawDescriptorBindingCount);
		}
		static inline void updateDebugDrawDescriptorSet(
			video::ILogicalDevice* device, video::IGPUDescriptorSet* debugDrawDS,
			asset::SBufferBinding<video::IGPUBuffer>&& skinInstanceOffsetList,
			asset::SBufferBinding<video::IGPUBuffer>&& skinInstanceJointCountXXXPrefixSum,
			asset::SBufferBinding<video::IGPUBuffer>&& aabbPool
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
				writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
				writes[i].info = infos+i;
			}
			infos[0] = skinInstanceOffsetList;
			infos[1] = skinInstanceJointCountXXXPrefixSum;
			infos[2] = aabbPool;
			device->updateDescriptorSets(DebugDrawDescriptorBindingCount,writes,0u,nullptr);
		}
		struct DebugPushConstants
		{
			core::matrix4SIMD viewProjectionMatrix;
			core::vector4df_SIMD aabbColor;
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
			cmdbuf->pushConstants(layout,asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(DebugPushConstants),&pushConstants);
			cmdbuf->drawIndexed(IndexCount,totalJointCount,0u,0u,0u);
		}

		//
		static inline constexpr uint32_t CacheUpdateDescriptorBindingCount = 2u;
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createCacheUpdateDescriptorSetLayout(video::ILogicalDevice* device, asset::ISpecializedShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[CacheUpdateDescriptorBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,CacheUpdateDescriptorBindingCount,asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createGPUDescriptorSetLayout(bindings,bindings+CacheUpdateDescriptorBindingCount);
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
			descSets.cacheUpdate = device->createGPUDescriptorSet(pool.get(),std::move(layouts[0]));
			descSets.debugDraw = device->createGPUDescriptorSet(pool.get(),std::move(layouts[1]));
			return descSets;
		}
	protected:
		ISkinInstanceCacheManager(
			core::smart_refctd_ptr<video::ILogicalDevice>&& _device,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& _cacheUpdate,
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>&& _debugDrawRenderpassIndependent,
			core::smart_refctd_ptr<video::IGPUBuffer>&& _initialTimestampBuffer,
			core::smart_refctd_ptr<video::IGPUBuffer>&& _debugIndexBuffer
		) : m_device(std::move(_device)), m_cacheUpdate(std::move(_cacheUpdate)),
			m_debugDrawRenderpassIndependent(std::move(_debugDrawRenderpassIndependent)),
			m_initialTimestampBuffer(std::move(_initialTimestampBuffer)),
			m_debugIndexBuffer(std::move(_debugIndexBuffer)),
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
		core::smart_refctd_ptr<video::IGPUBuffer> m_debugIndexBuffer;
		constexpr static inline auto IndexCount = 24u;
};

} // end namespace nbl::scene

#endif

