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
// TODO: split into ITT and ITTM because no need for multiple pipeline copies for multiple TTs
class ITransformTreeManager : public virtual core::IReferenceCounted
{
public:
    using node_t = uint32_t;
    _NBL_STATIC_INLINE_CONSTEXPR node_t invalid_node = video::IPropertyPool::invalid_index;

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
        relative_transform_t, modified_stamp_t,
        global_transform_t, recomputed_stamp_t>;

    struct RelativeTransformModificationRequest
    {
    public:
        enum E_TYPE : uint32_t
        {
            ET_OVERWRITE = 0u,  // exchange the value `This(vertex)`
            ET_CONCATENATE_AFTER = 1u,  // apply transform after `This(Previous(vertex))`
            ET_CONCATENATE_BEFORE = 2u,  // apply transform before `Previous(This(vertex))`
            ET_WEIGHTED_ACCUMULATE = 3u,  // add to existing value `(Previous+This)(vertex)`
            ET_COUNT
        };
        RelativeTransformModificationRequest(const E_TYPE type, const core::matrix3x4SIMD& _weightedModification)
            : storage(_weightedModification)
        {
            constexpr uint32_t log2ET_COUNT = 2u;
            static_assert(ET_COUNT <= (0x1u << log2ET_COUNT), "Need to rewrite the type encoding routine!");

            uint32_t typeBits[log2ET_COUNT];
            for(uint32_t i = 0u; i < log2ET_COUNT; i++)
                typeBits[i] = (type >> i) & 0x1u;

            // stuff the bits into x and z components of scale (without a rotation)
            reinterpret_cast<uint32_t&>(storage.rows[0].x) |= typeBits[0];
            reinterpret_cast<uint32_t&>(storage.rows[2].z) |= typeBits[1];
        }
        RelativeTransformModificationRequest(const E_TYPE type, const core::matrix3x4SIMD& _modification, const float weight)
            : RelativeTransformModificationRequest(type, _modification * weight) {}

        inline E_TYPE getType() const
        {
            uint32_t retval = reinterpret_cast<const uint32_t&>(storage.rows[0].x) & 0x1u;
            retval |= (reinterpret_cast<const uint32_t&>(storage.rows[2].z) & 0x1u) << 1u;
            return static_cast<E_TYPE>(retval);
        }

    private:
        core::matrix3x4SIMD storage;
    };

    // creation
    static inline core::smart_refctd_ptr<ITransformTreeManager> create(video::IVideoDriver* _driver, asset::SBufferRange<video::IGPUBuffer>&& memoryBlock, core::allocator<uint8_t>&& alloc = core::allocator<uint8_t>())
    {
        const auto reservedSize = video::IPropertyPool::getReservedSize(property_pool_t::calcApproximateCapacity(memoryBlock.size));
        auto reserved = std::allocator_traits<core::allocator<uint8_t>>::allocate(alloc, reservedSize);
        if(!reserved)
            return nullptr;

        auto retval = create(_driver, std::move(memoryBlock), reserved, std::move(alloc));
        if(!retval)
            std::allocator_traits<core::allocator<uint8_t>>::deallocate(alloc, reserved, reservedSize);

        return retval;
    }
    // if this method fails to create the pool, the callee must free the reserved memory themselves, also the reserved pointer must be compatible with the allocator so it can free it
    static inline core::smart_refctd_ptr<ITransformTreeManager> create(video::IVideoDriver* _driver, asset::SBufferRange<video::IGPUBuffer>&& memoryBlock, void* reserved, core::allocator<uint8_t>&& alloc = core::allocator<uint8_t>())
    {
        auto _nodeStorage = property_pool_t::create(std::move(memoryBlock), reserved, std::move(alloc));
        if(!_nodeStorage)
            return nullptr;

        auto* ttm = new ITransformTreeManager(_driver, std::move(_nodeStorage));
        return core::smart_refctd_ptr<ITransformTreeManager>(ttm, core::dont_grab);
    }

    //
    inline const auto* getNodePropertyPool() const { return m_nodeStorage.get(); }

    //
    inline asset::SBufferRange<video::IGPUBuffer> getGlobalTransformationBufferRange() const
    {
        asset::SBufferRange<video::IGPUBuffer> retval = {m_nodeStorage->getPropertyOffset(global_transform_prop_ix), m_nodeStorage->getCapacity() * sizeof(global_transform_t), m_nodeStorage->getMemoryBlock().buffer};
        return retval;
    }

    // need to at least initialize with the parent node property with the recompute and update timestamps at 0xfffffffeu and 0xffffffffu respectively
    // but a function with optional relative transform would be nice
    // need our own compute shader to initialize the properties ;(
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
        m_nodeStorage->freeProperties(begin, end);
    }
    //
    inline void clearNodes()
    {
        m_nodeStorage->freeAllProperties();
    }

    // TODO: make all these functions take a pipeline barrier type (future new API) with default being a full barrier
    template<typename... Args>
    inline void updateLocalTransforms(Args&&... args)
    {
        soleUpdateOrFusedRecompute_impl(m_updatePipeline.get(), std::forward<Args>(args)...);
    }
    //
    void recomputeGlobalTransforms(const asset::SBufferBinding<video::IGPUBuffer>& dispatchIndirectParameters, const asset::SBufferBinding<video::IGPUBuffer>& nodeIDBuffer)
    {
        // TODO: do it properly
        auto out = getGlobalTransformationBufferRange();
        m_driver->copyBuffer(m_nodeStorage->getMemoryBlock().buffer.get(), out.buffer.get(), m_nodeStorage->getPropertyOffset(1u), out.offset, out.size);
    }
    //
    template<typename... Args>
    inline void updateAndRecomputeTransforms(Args&&... args)
    {
        soleUpdateOrFusedRecompute_impl(m_updateAndRecomputePipeline.get(), std::forward<Args>(args)...);
    }

    //
    auto transferGlobalTransforms(const node_t* begin, const node_t* end, const asset::SBufferBinding<video::IGPUBuffer>& outputBuffer, const std::chrono::steady_clock::time_point& maxWaitPoint = video::GPUEventWrapper::default_wait())
    {
        video::CPropertyPoolHandler::TransferRequest request;
        request.download = true;
        request.pool = m_nodeStorage.get();
        request.indices = {begin, end};
        request.propertyID = 3u;
        //m_nodeStorage->transferProperties();
        assert(false);  // TODO: Need a transfer to GPU mem
    }
    //auto downloadGlobalTransforms()

protected:
    ITransformTreeManager(video::IVideoDriver* _driver, core::smart_refctd_ptr<property_pool_t>&& _nodeStorage)
        : m_driver(_driver), m_nodeStorage(std::move(_nodeStorage))
    {
        // TODO: the ComputePipeline for update,recompute and combined update&recompute
    }
    ~ITransformTreeManager()
    {
        // everything drops itself automatically
    }

    void soleUpdateOrFusedRecompute_impl(
        const video::IGPUComputePipeline* pipeline,
        const asset::SBufferBinding<video::IGPUBuffer>& dispatchIndirectParameters,
        const asset::SBufferBinding<video::IGPUBuffer>& nodeIDBuffer,  // first uint in the nodeIDBuffer is used to denote how many requests we have
        const asset::SBufferBinding<video::IGPUBuffer>& modificationRequestBuffer,
        const asset::SBufferBinding<video::IGPUBuffer>& modificationRequestTimestampBuffer)
    {
        // TODO: first a dispatch to sort the modification requests and timestamps according to node frequency
        m_driver->bindComputePipeline(pipeline);
        assert(false);  // TODO: get a descriptor set to populate with our input buffers (plus indirect dispatch buffer + nodeIDBuffer if pipeline==m_updateAndRecomputePipeline)
        const video::IGPUDescriptorSet* descSets[] = {m_transformHierarchyDS.get(), nullptr};
        m_driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline->getLayout(), 0u, 2u, descSets, nullptr);
        m_driver->dispatchIndirect(dispatchIndirectParameters.buffer.get(), dispatchIndirectParameters.offset);
        // TODO: pipeline barrier for UBO, SSBO and TBO and if pipeline==m_updatePipeline then COMMAND_BIT too
    }

    video::IVideoDriver* m_driver;
    core::smart_refctd_ptr<property_pool_t> m_nodeStorage;
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_updatePipeline, m_recomputePipeline, m_updateAndRecomputePipeline;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> m_transformHierarchyDS;
    // TODO: do we keep a contiguous `node_t` array in-case we want to shortcut to full tree reevaluation when the number of relative transform modification requsts > totalNodes*ratio (or overflows the temporary buffer we've provided) ?
    /** Ideal O(1) insertion and erasure
		* Add: new nodes using pool allocator, add the node_t references to the back of the contiguous array (increment atomic) and record where the reference is (offset into contiguous) as an additional property of the node
		* Remove: lookup the contiguous offset property for the removed node, decremenet contiguous array size atomic and save the return value as the reference to be swapped, swap the erased reference with the one to be swapped, dereference the swapped reference and update its pointer to contiguous 
		**/
};

}  // end namespace scene
}  // end namespace nbl

#endif
