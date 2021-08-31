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
#define uvec4 core::vectorSIMDu32
#include "nbl/builtin/glsl/transform_tree/relative_transform_modification.glsl"
#undef uvec4
#undef uint

class ITransformTreeManager : public virtual core::IReferenceCounted
{
	public:
		using node_t = uint32_t;
		_NBL_STATIC_INLINE_CONSTEXPR node_t invalid_node = video::IPropertyPool::invalid;

		using timestamp_t = video::IGPUAnimationLibrary::timestamp_t;
		// two timestamp values are reserved for initialization
		_NBL_STATIC_INLINE_CONSTEXPR timestamp_t min_timestamp = 0u;
		_NBL_STATIC_INLINE_CONSTEXPR timestamp_t max_timestamp = 0xfffffffdu;

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t parent_prop_ix = 0u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t global_transform_prop_ix = 3u;
	private:
		using parent_t = node_t;
		using relative_transform_t = core::matrix3x4SIMD;
		using modified_stamp_t = timestamp_t;
		using global_transform_t = core::matrix3x4SIMD;
		using recomputed_stamp_t = timestamp_t;

	public:
		using property_pool_t = video::CPropertyPool<core::allocator,
			parent_t,
			relative_transform_t,modified_stamp_t,
			global_transform_t,recomputed_stamp_t
		>;

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
			// TODO: create the pipelines for alloc,update,recompute and combined update&recompute in the constructor

			auto* ttm = new ITransformTreeManager(std::move(device));
            return core::smart_refctd_ptr<ITransformTreeManager>(ttm,core::dont_grab);
        }

		struct AllocationRequest
		{
			video::StreamingTransientDataBufferMT<>* upBuff;
			// must be in recording state
			video::IGPUCommandBuffer* cmdbuf;
			video::IGPUFence* fence;
			ITransformTree* tree;
			core::SRange<node_t> outNodes;
			const parent_t*	parents;
			// if null we don't set the relativeTransforms
			const relative_transform_t*	relativeTransforms = nullptr;
		};
		inline bool addNodes(const AllocationRequest& request, const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait())
		{
			if (!request.tree)
				return false;
			auto* pool = request.tree->getNodePropertyPool();
			if (request.outNodes.size()>pool->getFree())
				return false;

			pool->allocateProperties(request.outNodes.begin(),request.outNodes.end());
			// TODO: Need to run a `CPropertyPoolHandler`-like transfer compute shader to transfer `parent`, intiailize timestamps, and transfer/initialize relativeTransforms
			// need to at least initialize with the parent node property with the recompute and update timestamps at 0xfffffffeu and 0xffffffffu respectively
			assert(false);
			return true;
		}
		// TODO: utilities for adding root nodes, adding skeleton node instances, etc.
		 
		//
		inline void removeNodes(ITransformTree* tree, const node_t* begin, const node_t* end)
		{
			// If we start wanting a contiguous range to be maintained, this will need to change
			tree->getNodePropertyPool()->freeProperties(begin,end);
		}

		// TODO: make all these functions take a pipeline barrier type (future new API) with default being a full barrier
		struct ParamsBase
		{
			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
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
				uint32_t srcQueueFamilyIndex;
				uint32_t dstQueueFamilyIndex;
				asset::E_PIPELINE_STAGE_FLAGS dstStages = asset::EPSF_ALL_COMMANDS_BIT;
				asset::E_ACCESS_FLAGS dstAccessMask = asset::EAF_ALL_ACCESSES_BIT_DEVSH;
			} finalBarrier = {};
			asset::SBufferBinding<video::IGPUBuffer> nodeIDBuffer; // first uint in the buffer tells us how many requests we have
		};
		struct LocalTransformUpdateParams : ParamsBase
		{
			video::IGPUFence fence; // for signalling when to drop a temporary descriptor set
			asset::SBufferBinding<video::IGPUBuffer> modificationRequestBuffer;
			asset::SBufferBinding<video::IGPUBuffer> modificationRequestTimestampBuffer;
			bool duplicateNodeReferences = true; // whether the list of updates could contain multiple updates to the same node
		};
		inline void updateLocalTransforms(const LocalTransformUpdateParams& params)
		{
			soleUpdateOrFusedRecompute_impl(m_updatePipeline.get(),params);
		}
		//
		void recomputeGlobalTransforms(const ParamsBase& params)
		{
			auto* cmdbuf = params.cmdbuf;
			cmdbuf->bindComputePipeline(m_recomputePipeline.get());
			const video::IGPUDescriptorSet* descSets[] = { params.tree->getNodePropertyDescriptorSet() };
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_recomputePipeline->getLayout(),0u,1u,descSets);
			lastDispatch(m_recomputePipeline.get(),params);
		}
		//
		inline void updateAndRecomputeTransforms(const LocalTransformUpdateParams& params)
		{
			soleUpdateOrFusedRecompute_impl(m_updateAndRecomputePipeline.get(),params);
		}

		static inline constexpr uint32_t WorkgroupSize = 256u;
	protected:
		ITransformTreeManager(core::smart_refctd_ptr<video::ILogicalDevice>&& _device) : m_device(_device)
		{
			// TODO: take the ComputePipelines for alloc,update,recompute and combined update&recompute in the constructor
		}
		~ITransformTreeManager()
		{
			// everything drops itself automatically
		}

		void soleUpdateOrFusedRecompute_impl(const video::IGPUComputePipeline* pipeline, const LocalTransformUpdateParams& params)
		{
			auto* cmdbuf = params.cmdbuf;
			if (params.duplicateNodeReferences)
			{
				// TODO: first a dispatch to stable sort the SoA requests according to nodeID
				// could histogram the nodes first (maybe sort the nodes by amount of references)
				// prefix sum/scan and get my premade offsets for contiguous sublists
				// scatter and local sort my requests
				// process all nodes' requests easily with 1 invocation : 1 node
				assert(false);
			}
			// TODO: get a descriptor set to populate with our input buffers (plus indirect dispatch buffer + nodeIDBuffer if pipeline==m_updateAndRecomputePipeline)
			assert(false);
			core::smart_refctd_ptr<video::IGPUDescriptorSet> tempDS;
			// TOOD: do what CPropertyPoolHandler does and fill tempDS from some sort of reclaimable cache
			const video::IGPUDescriptorSet* descSets[] = { params.tree->getNodePropertyDescriptorSet(),tempDS.get() };
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,pipeline->getLayout(),0u,2u,descSets,nullptr);

			lastDispatch(pipeline,params);

			// TODO: put the tempDS on the deferred free list of IDescriptorSetCache
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
				asset::EDF_NONE,0u,nullptr,4u,bufferBarriers,0u,nullptr
			);
		}

		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_allocatePipeline,m_updatePipeline,m_recomputePipeline,m_updateAndRecomputePipeline;
};

} // end namespace nbl::scene

#endif

