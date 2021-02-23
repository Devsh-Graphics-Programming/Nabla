// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_TREE_TRANSFORM_MANAGER_H_INCLUDED__
#define __NBL_SCENE_I_TREE_TRANSFORM_MANAGER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/video/video.h"

namespace nbl
{
namespace scene
{


class ITransformTreeManager : public virtual core::IReferenceCounted
{
	public:
		using node_t = uint32_t;
		_NBL_STATIC_INLINE_CONSTEXPR node_t invalid_node = video::IPropertyPool::invalid_index;

		using timestamp_t = video::IGPUAnimationLibrary::timestamp_t;

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

		// creation
		static inline core::smart_refctd_ptr<ITransformTreeManager> create(video::IVideoDriver* _driver, asset::SBufferRange<video::IGPUBuffer>&& memoryBlock, core::allocator<uint8_t>&& alloc = core::allocator<uint8_t>())
		{
			const auto reservedSize = video::IPropertyPool::getReservedSize(property_pool_t::calcApproximateCapacity(memoryBlock.size));
			auto reserved = std::allocator_traits<core::allocator<uint8_t>>::allocate(alloc,reservedSize);
			if (!reserved)
				return nullptr;

			auto retval = create(_driver,std::move(memoryBlock),reserved,std::move(alloc));
			if (!retval)
				std::allocator_traits<core::allocator<uint8_t>>::deallocate(alloc,reserved,reservedSize);

			return retval;
		}
		// if this method fails to create the pool, the callee must free the reserved memory themselves, also the reserved pointer must be compatible with the allocator so it can free it
        static inline core::smart_refctd_ptr<ITransformTreeManager> create(video::IVideoDriver* _driver, asset::SBufferRange<video::IGPUBuffer>&& memoryBlock, void* reserved, core::allocator<uint8_t>&& alloc=core::allocator<uint8_t>())
        {
			auto _nodeStorage = property_pool_t::create(std::move(memoryBlock),reserved,std::move(alloc));
			if (!_nodeStorage)
				return nullptr;

			auto* ttm = new ITransformTreeManager(_driver,std::move(_nodeStorage));
            return core::smart_refctd_ptr<ITransformTreeManager>(ttm,core::dont_grab);
        }
		
		//
		inline const auto* getNodePropertyPool() const {return m_nodeStorage.get();}

		//
		inline asset::SBufferRange<video::IGPUBuffer> getGlobalTransformationBufferRange() const
		{
			asset::SBufferRange<video::IGPUBuffer> retval = {m_nodeStorage->getPropertyOffset(global_transform_prop_ix),m_nodeStorage->getCapacity()*sizeof(global_transform_t),m_nodeStorage->getMemoryBlock().buffer};
			return retval;
		}
#if 0
		//
		struct AllocationRequest
		{
			core::SRange<node_t> outNodes;
			const parent_t*	parents;
			const relative_transform_t*	relativeTransforms;
			// what to do about timestamps?
		};
		template<typename ParentNodeIt>
		inline void addNodes(CPropertyPoolHandler* propertyPoolHandler, node_t* nodesBegin, node_t* nodesEnd, ParentNodeIt parentsBegin, const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait())
		{
			if (std::distance(nodesBegin,nodesEnd)>m_nodeStorage->getFree())
				return;

			m_nodeStorage->allocateProperties(nodesBegin,nodesEnd);
			assert(false); // TODO
		}
		// TODO: utilities for adding root nodes, adding skeleton node instances, etc.
		// should we just do it ourselves with a shader? (set correct timestamps so global gets recomputed)
#endif
		//
		inline void removeNodes(const node_t* begin, const node_t* end)
		{
			m_nodeStorage->freeProperties(begin,end);
		}
		//
		inline void clearNodes()
		{
			m_nodeStorage->freeAllProperties();
		}

		// TODO: make all these functions take a pipeline barrier type (future new API) with default being a full barrier
		void updateLocalTransforms(const asset::SBufferBinding<video::IGPUBuffer>& dispatchIndirectParameters, const asset::SBufferBinding<video::IGPUBuffer>& requestBuffer) //do we take a per-request nodeID buffer too?
		{
			assert(false); // TODO
		}
		//
		void recomputeGlobalTransforms(/*dispatchIndirectParams,nodeList*/)
		{
			// TODO: do it properly
			auto out = getGlobalTransformationBufferRange();
			m_driver->copyBuffer(m_nodeStorage->getMemoryBlock().buffer.get(),out.buffer.get(),m_nodeStorage->getPropertyOffset(1u),out.offset,out.size);
		}
		//
		void updateAndRecomputeGlobalTransforms(/*Same args as `updateLocalTransforms` and `recomputeGlobalTransforms`*/)
		{
			assert(false); // TODO
		}

		//
		auto transferGlobalTransforms(const node_t* begin, const node_t* end, const asset::SBufferBinding<video::IGPUBuffer>& outputBuffer, const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait())
		{
			video::CPropertyPoolHandler::TransferRequest request;
			request.download = true;
			request.pool = m_nodeStorage.get();
			request.indices = {begin,end};
			request.propertyID = 3u;
			//m_nodeStorage->transferProperties();
			assert(false); // TODO: Need a transfer to GPU mem
		}
		//auto downloadGlobalTransforms()

	protected:
		ITransformTreeManager(video::IVideoDriver* _driver, core::smart_refctd_ptr<property_pool_t>&& _nodeStorage) : m_driver(_driver), m_nodeStorage(std::move(_nodeStorage))
		{
			// TODO: the ComputePipeline for update,recompute and combined update&recompute
		}
		~ITransformTreeManager()
		{
			//
		}

#if 0
		struct RootNodeParentIterator
		{
			inline RootNodeParentIterator& operator++()
			{
				//do nothing
				return *this;
			}
			inline RootNodeParentIterator operator++(int)
			{
				//do nothing
			}

			inline node_t operator*() const
			{
				return invalid_node;
			}

			//using iterator_category = typename std::iterator_traits::;
			//using difference_type = ptrdiff_t;
			using value_type = node_t;
		};
#endif
		video::IVideoDriver* m_driver;
		core::smart_refctd_ptr<property_pool_t> m_nodeStorage;
};


} // end namespace scene
} // end namespace nbl

#endif

