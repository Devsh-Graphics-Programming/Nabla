// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_TREE_TRANSFORM_MANAGER_H_INCLUDED__
#define __NBL_SCENE_I_TREE_TRANSFORM_MANAGER_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/video/declarations.h"

#include "nbl/core/definitions.h"

#include "nbl/scene/ITransformTree.h"

namespace nbl::scene
{

//
#define uint uint32_t
#define int int32_t
#define uvec4 core::vectorSIMDu32
#include "nbl/builtin/glsl/transform_tree/relative_transform_modification.glsl"
#include "nbl/builtin/glsl/transform_tree/modification_request_range.glsl"
#undef uvec4
#undef int
#undef uint

class ITransformTreeManager : public virtual core::IReferenceCounted
{
	public:
		struct RelativeTransformModificationRequest : nbl_glsl_transform_tree_relative_transform_modification_t
		{
			public:
				enum E_TYPE : uint32_t
				{
					ET_OVERWRITE=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_OVERWRITE_, // exchange the value, `This(vertex)`
					ET_CONCATENATE_AFTER=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_AFTER_, // apply transform after, `This(Previous(vertex))`
					ET_CONCATENATE_BEFORE=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_BEFORE_, // apply transform before, `Previous(This(vertex))`
					ET_WEIGHTED_ACCUMULATE=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_WEIGHTED_ACCUMULATE_, // add to existing value, `(Previous+This)(vertex)`
					ET_COUNT=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_COUNT_
				};
				RelativeTransformModificationRequest() = default;
				RelativeTransformModificationRequest(const E_TYPE type, const core::matrix3x4SIMD& _preweightedModification)
				{
					constexpr uint32_t log2ET_COUNT = 2u;
					static_assert(ET_COUNT<=(0x1u<<log2ET_COUNT),"Need to rewrite the type encoding routine!");
				
					//
					*reinterpret_cast<core::matrix3x4SIMD*>(data) = _preweightedModification;

					// stuff the bits into x and z components of scale (without a rotation) 
					// clear then bitwise-or
					data[0][0] &= 0xfffffffeu;
					data[0][0] |= type&0x1u;
					data[2][2] &= 0xfffffffeu;
					data[2][2] |= (type>>1u)&0x1u;
				}
				RelativeTransformModificationRequest(const E_TYPE type, const core::matrix3x4SIMD& _modification, const float weight) : RelativeTransformModificationRequest(type,_modification*weight) {}

				inline E_TYPE getType() const
				{
					return static_cast<E_TYPE>(nbl_glsl_transform_tree_relative_transform_modification_t_getType(*this));
				}
		};

		// creation
        static inline core::smart_refctd_ptr<ITransformTreeManager> create(core::smart_refctd_ptr<video::ILogicalDevice>&& device)
        {
			auto system = device->getPhysicalDevice()->getSystem();
			auto createShader = [&system,&device](auto uniqueString) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
			{
				auto glsl = system->loadBuiltinData<decltype(uniqueString)>();
				auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl),asset::IShader::buffer_contains_glsl_t{});
				auto shader = device->createGPUShader(asset::IGLSLCompiler::createOverridenCopy(cpushader.get(),"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n",WorkgroupSize));
				return device->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});
			};

			auto updateRelativeSpec = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/transform_tree/relative_transform_update.comp")());
			auto recomputeGlobalSpec = createShader(NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/transform_tree/global_transform_update.comp")());
			if (!updateRelativeSpec || !recomputeGlobalSpec)
				return nullptr;

			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> sharedDsLayout;
			{
				video::IGPUDescriptorSetLayout::SBinding bnd[2];
				bnd[0].binding = 0u;
				bnd[0].count = 1u;
				bnd[0].type = asset::EDT_STORAGE_BUFFER;
				bnd[0].stageFlags = video::IGPUSpecializedShader::ESS_COMPUTE;
				bnd[0].samplers = nullptr;
				bnd[1] = bnd[0];
				bnd[1].binding = 1u;
				sharedDsLayout = device->createGPUDescriptorSetLayout(bnd,bnd+2);
			}
			asset::ISpecializedShader::E_SHADER_STAGE stageAccessFlags[ITransformTree::property_pool_t::PropertyCount];
			std::fill_n(stageAccessFlags,ITransformTree::property_pool_t::PropertyCount,asset::ISpecializedShader::ESS_COMPUTE);
			auto poolLayout = ITransformTree::createDescriptorSetLayout(device.get(),stageAccessFlags);
			
			auto updateRelativeLayout = device->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(poolLayout),core::smart_refctd_ptr(sharedDsLayout));
			auto recomputeGlobalLayout = device->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(poolLayout),core::smart_refctd_ptr(sharedDsLayout));

			auto updateRelativePpln = device->createGPUComputePipeline(nullptr,std::move(updateRelativeLayout),std::move(updateRelativeSpec));
			auto recomputeGlobalPpln = device->createGPUComputePipeline(nullptr,std::move(recomputeGlobalLayout),std::move(recomputeGlobalSpec));
			if (!updateRelativePpln || !recomputeGlobalPpln)
				return nullptr;

			// TODO: after BaW
			core::smart_refctd_ptr<video::IGPUComputePipeline> updateAndRecomputePpln;
			
			// TODO: if we decide to invalidate all cmdbuffs used for updates (make them non reusable), then we can use the ECF_NONE flag
			auto descPool = device->createDescriptorPoolForDSLayouts(
				video::IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT,
				&sharedDsLayout.get(),&sharedDsLayout.get()+1u,
				&DescriptorCacheSize
			);
			auto descCache = core::make_smart_refctd_ptr<DescriptorSetCache>(device.get(),std::move(descPool),std::move(sharedDsLayout));

			auto* ttm = new ITransformTreeManager(
				std::move(device),std::move(descCache),
				std::move(updateRelativePpln),
				std::move(recomputeGlobalPpln),
				std::move(updateAndRecomputePpln)
			);
            return core::smart_refctd_ptr<ITransformTreeManager>(ttm,core::dont_grab);
        }

		//
		struct AllocationRequest
		{
			video::CPropertyPoolHandler* poolHandler;
			video::StreamingTransientDataBufferMT<>* upBuff;
			// must be in recording state
			video::IGPUCommandBuffer* cmdbuf;
			video::IGPUFence* fence;
			ITransformTree* tree;
			core::SRange<ITransformTree::node_t> outNodes = { nullptr, nullptr };
			// if null we set these properties to defaults (no parent and identity transform)
			const ITransformTree::parent_t* parents = nullptr;
			const ITransformTree::relative_transform_t* relativeTransforms = nullptr;
			system::logger_opt_ptr logger = nullptr;
		};
		inline bool addNodes(const AllocationRequest& request, const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait())
		{
			if (!request.poolHandler || !request.upBuff || !request.cmdbuf || !request.fence || !request.tree)
				return false;

			auto* pool = request.tree->getNodePropertyPool();
			if (request.outNodes.size()>pool->getFree())
				return false;

			pool->allocateProperties(request.outNodes.begin(),request.outNodes.end());
			const core::matrix3x4SIMD IdentityTransform;

			constexpr auto TransferCount = 4u;
			video::CPropertyPoolHandler::TransferRequest transfers[TransferCount];
			for (auto i=0u; i<TransferCount; i++)
			{
				transfers[i].pool = pool;
				transfers[i].elementCount = request.outNodes.size();
				transfers[i].srcAddresses = nullptr;
				transfers[i].dstAddresses = request.outNodes.begin();
				transfers[i].device2device = false;
			}
			transfers[0].propertyID = ITransformTree::parent_prop_ix;
			transfers[0].flags = request.parents ? video::CPropertyPoolHandler::TransferRequest::EF_NONE:video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[0].source = request.parents ? request.parents:&ITransformTree::invalid_node;
			transfers[1].propertyID = ITransformTree::relative_transform_prop_ix;
			transfers[1].flags = request.relativeTransforms ? video::CPropertyPoolHandler::TransferRequest::EF_NONE:video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[1].source = request.relativeTransforms ? request.relativeTransforms:&IdentityTransform;
			transfers[2].propertyID = ITransformTree::modified_stamp_prop_ix;
			transfers[2].flags = video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[2].source = &ITransformTree::initial_modified_timestamp;
			transfers[3].propertyID = ITransformTree::recomputed_stamp_prop_ix;
			transfers[3].flags = video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[3].source = &ITransformTree::initial_recomputed_timestamp;
			return request.poolHandler->transferProperties(request.upBuff,nullptr,request.cmdbuf,request.fence,transfers,transfers+TransferCount,request.logger,maxWaitPoint).transferSuccess;
		}
		// TODO: utility for adding skeleton node instances, etc.
		 
		//
		inline void removeNodes(ITransformTree* tree, const ITransformTree::node_t* begin, const ITransformTree::node_t* end)
		{
			// If we start wanting a contiguous range to be maintained, this will need to change
			tree->getNodePropertyPool()->freeProperties(begin,end);
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

			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			// for signalling when to drop a temporary descriptor set
			video::IGPUFence* fence;
			ITransformTree* tree;
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
			struct BarrierParams
			{
				// TODO: what to set queue family indices to if we don't plan on a transfer by default?
				uint32_t srcQueueFamilyIndex;
				uint32_t dstQueueFamilyIndex;
				asset::E_PIPELINE_STAGE_FLAGS dstStages = asset::EPSF_ALL_COMMANDS_BIT;
				asset::E_ACCESS_FLAGS dstAccessMask = asset::EAF_ALL_ACCESSES_BIT_DEVSH;
			} finalBarrier = {};
			system::logger_opt_ptr logger = nullptr;
		};

		//
		struct LocalTransformUpdateParams : ParamsBase
		{
			// first uint in the buffer tells us how many ModificationRequestRanges we have
			// second uint in the buffer tells us how many total requests we have
			// rest is filled wtih ModificationRequestRange
			asset::SBufferBinding<video::IGPUBuffer> requestRanges;
			// this one is filled with RelativeTransformModificationRequest
			asset::SBufferBinding<video::IGPUBuffer> modificationRequests;
		};
		inline bool updateLocalTransforms(const LocalTransformUpdateParams& params)
		{
			return soleUpdateOrFusedRecompute_impl<2u>(m_updatePipeline.get(),params,{params.requestRanges,params.modificationRequests});
		}

		//
		struct GlobalTransformUpdateParams : ParamsBase
		{
			// first uint in the buffer tells us how many nodes to update we have
			asset::SBufferBinding<video::IGPUBuffer> nodeIDs; // imo it should be SBufferRange
		};
		inline bool recomputeGlobalTransforms(const GlobalTransformUpdateParams& params)
		{
			return soleUpdateOrFusedRecompute_impl<1u>(m_recomputePipeline.get(),params,{params.nodeIDs});
		}

		//
		struct RelativeTransformUpdateAndGlobalTransformUpdateParams : LocalTransformUpdateParams
		{
			// first uint in the buffer tells us how many nodes to update we have
			asset::SBufferRange<video::IGPUBuffer> nodeIDs;
		};
		inline bool updateAndRecomputeTransforms(const RelativeTransformUpdateAndGlobalTransformUpdateParams& params)
		{
			// TODO: after BaW, for now just use `updateLocalTransforms` and `recomputeGlobalTransforms` in order
			assert(false);
			return false;// commonDispatch(m_updateAndRecomputePipeline.get(), params, video::IDescriptorSetCache::invalid_index);
		}

		static inline constexpr uint32_t WorkgroupSize = 256u;

	protected:
		static inline constexpr auto DescriptorCacheSize = 16u;
		// TODO: investigate using Push Descriptors for this
		class DescriptorSetCache : public video::IDescriptorSetCache
		{
			public:
				using IDescriptorSetCache::IDescriptorSetCache;

				static constexpr inline auto SharedBindingCount = 3u;

				uint32_t acquireSet(video::ILogicalDevice* device, const asset::SBufferRange<video::IGPUBuffer>* buffers, uint32_t count)
				{
					auto retval = IDescriptorSetCache::acquireSet();
					if (retval==IDescriptorSetCache::invalid_index)
						return IDescriptorSetCache::invalid_index;
					video::IGPUDescriptorSet* set = IDescriptorSetCache::getSet(retval);

					video::IGPUDescriptorSet::SWriteDescriptorSet writes[SharedBindingCount];
					video::IGPUDescriptorSet::SDescriptorInfo infos[SharedBindingCount];
					for (auto i=0u; i<count; i++)
					{
						infos[i].desc = buffers[i].buffer;
						infos[i].buffer.offset = buffers[i].offset;
						infos[i].buffer.size = buffers[i].size;
						writes[i].dstSet = set;
						writes[i].binding = i;
						writes[i].arrayElement = 0u;
						writes[i].count = 1u;
						writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
						writes[i].info = infos+i;
					}
					device->updateDescriptorSets(count,writes,0u,nullptr);

					return retval;
				}
		};
		ITransformTreeManager(
			core::smart_refctd_ptr<video::ILogicalDevice>&& _device,
			core::smart_refctd_ptr<DescriptorSetCache>&& _dsCache,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& _updatePipeline,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& _recomputePipeline,
			core::smart_refctd_ptr<video::IGPUComputePipeline>&& _updateAndRecomputePipeline
		) : m_device(std::move(_device)), m_dsCache(std::move(_dsCache)),
			m_updatePipeline(std::move(_updatePipeline)),
			m_recomputePipeline(std::move(_recomputePipeline)),
			m_updateAndRecomputePipeline(std::move(_updateAndRecomputePipeline))
		{
		}
		~ITransformTreeManager()
		{
			// everything drops itself automatically
		}
		
		template<uint32_t N>
		bool soleUpdateOrFusedRecompute_impl(const video::IGPUComputePipeline* pipeline, const ParamsBase& params, const std::array<asset::SBufferBinding<video::IGPUBuffer>,N>& bufferBindings)
		{
			auto* cmdbuf = params.cmdbuf;

			auto dsix = m_dsCache->acquireSet();
			if (dsix==video::IDescriptorSetCache::invalid_index)
			{
				params.logger.log("CPropertyPoolHandler: Failed to acquire descriptor set!",system::ILogger::ELL_ERROR);
				return false;
			}
			video::IGPUDescriptorSet* tempDS = m_dsCache->getSet(dsix);
			{
				constexpr auto MaxBindingCount = 2u;

				video::IGPUDescriptorSet::SDescriptorInfo info[MaxBindingCount];
				for (auto i=0u; i<N; i++)
				{
					info[i].desc = bufferBindings[i].buffer;
					info[i].buffer.offset = bufferBindings[i].offset;
					info[i].buffer.size = video::IGPUDescriptorSet::SDescriptorInfo::SBufferInfo::WholeBuffer;
				}
				video::IGPUDescriptorSet::SWriteDescriptorSet w[MaxBindingCount];
				for (auto i=0u; i<MaxBindingCount; i++)
				{
					w[i].arrayElement = 0u;
					w[i].binding = i;
					w[i].count = 1u;
					w[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					w[i].info = info+std::min(i,N-1u);
					w[i].dstSet = tempDS;
				}
				m_device->updateDescriptorSets(MaxBindingCount,w,0u,nullptr);
			}
			const video::IGPUDescriptorSet* descSets[] = { params.tree->getNodePropertyDescriptorSet(),tempDS };
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,pipeline->getLayout(),0u,2u,descSets,nullptr);

			lastDispatch(pipeline,params);

			m_dsCache->releaseSet(m_device.get(),core::smart_refctd_ptr<video::IGPUFence>(params.fence),dsix);
			return true;
		}

		void lastDispatch(const video::IGPUComputePipeline* pipeline, const ParamsBase& params)
		{
			auto* cmdbuf = params.cmdbuf;
			cmdbuf->bindComputePipeline(pipeline);
			
			if (params.dispatchIndirect.buffer)
				cmdbuf->dispatchIndirect(params.dispatchIndirect.buffer,params.dispatchIndirect.offset);
			else
				cmdbuf->dispatch((params.dispatchDirect.nodeCount-1u)/WorkgroupSize+1u,1u,1u); // TODO: @Przemog would really like that dispatch factorization function
				

			// we always add our own stage and access flags, simply to have up to date data available for the next time we run the shader
			uint32_t barrierCount = 0u;
			video::IGPUCommandBuffer::SBufferMemoryBarrier bufferBarriers[ITransformTree::property_pool_t::PropertyCount-1u];
			auto setUpBarrier = [&](uint32_t prop_ix)
			{
				auto& bufBarrier = bufferBarriers[barrierCount++];
				bufBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
				bufBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(params.finalBarrier.dstAccessMask|asset::EAF_SHADER_READ_BIT|asset::EAF_SHADER_WRITE_BIT);
				bufBarrier.srcQueueFamilyIndex = params.finalBarrier.srcQueueFamilyIndex;
				bufBarrier.dstQueueFamilyIndex = params.finalBarrier.dstQueueFamilyIndex;
				const auto& block = params.tree->getNodePropertyPool()->getPropertyMemoryBlock(prop_ix);
				bufBarrier.buffer = block.buffer;
				bufBarrier.offset = block.offset;
				bufBarrier.size = block.size;
			};
			// update is being done
			if (pipeline!=m_recomputePipeline.get())
			{
				setUpBarrier(ITransformTree::relative_transform_prop_ix);
				setUpBarrier(ITransformTree::modified_stamp_prop_ix);
			}
			// recomputation is being done
			if (pipeline!=m_updatePipeline.get())
			{
				setUpBarrier(ITransformTree::global_transform_prop_ix);
				setUpBarrier(ITransformTree::recomputed_stamp_prop_ix);
			}
			cmdbuf->pipelineBarrier(
				asset::EPSF_COMPUTE_SHADER_BIT,params.finalBarrier.dstStages|asset::EPSF_COMPUTE_SHADER_BIT,
				asset::EDF_NONE,0u,nullptr,barrierCount,bufferBarriers,0u,nullptr
			);
		}

		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IDescriptorSetCache> m_dsCache;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_updatePipeline,m_recomputePipeline,m_updateAndRecomputePipeline;
};

} // end namespace nbl::scene

#endif

