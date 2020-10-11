#ifndef __NBL_VIDEO_I_PROPERTY_POOL_H_INCLUDED__
#define __NBL_VIDEO_I_PROPERTY_POOL_H_INCLUDED__


#include "irr/asset/asset.h"

#include "IGPUBuffer.h"


namespace irr
{
namespace video
{


// property pool is inherently single threaded
class IPropertyPool : public core::IReferenceCounted
{
	public:
		using PropertyAddressAllocator = core::PoolAddressAllocatorST<uint32_t>;

        _IRR_STATIC_INLINE_CONSTEXPR auto invalid_index = PropertyAddressAllocator::invalid_address;

		//
		inline const asset::SBufferRange<IGPUBuffer>& getMemoryBlock() const { return memoryBlock; }

		//
		virtual uint32_t getPropertyCount() const =0;
		virtual size_t getPropertyOffset(uint32_t ix) const =0;
		virtual uint32_t getPropertySize(uint32_t ix) const =0;

        //
        inline uint32_t getAllocated() const
        {
            return indexAllocator.get_allocated_size();
        }

        //
        inline uint32_t getFree() const
        {
            return indexAllocator.get_free_size();
        }

        //
        inline uint32_t getCapacity() const
        {
            // special case allows us to use `get_total_size`, because the pool allocator has no added offsets
            return indexAllocator.get_total_size();
        }

        // allocate, indices need to be pre-initialized to `invalid_index`
        inline bool allocateProperties(uint32_t* outIndicesBegin, uint32_t* outIndicesEnd)
        {
            constexpr uint32_t unit = 1u;
            for (auto it=outIndicesBegin; it!=outIndicesEnd; it++)
            {
                auto& addr = *it;
                if (addr!=invalid_index)
                    continue;

                addr = indexAllocator.alloc_addr(unit,unit);
                if (addr==invalid_index)
                    return false;
            }
            return true;
        }

        //
        inline void freeProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd)
        {
            constexpr uint32_t unit = 1u;
            for (auto it=indicesBegin; it!=indicesEnd; it++)
            {
                auto& addr = *it;
                if (addr!=invalid_index)
                    indexAllocator.free_addr(addr,unit);
            }
        }

        //
        inline void freeAllProperties()
        {
            indexAllocator.reset();
        }

    protected:
        #define PROPERTY_ADDRESS_ALLOCATOR_ARGS 1u,capacity,1u
        static inline PropertyAddressAllocator::size_type getReservedSize(uint32_t capacity)
        {
            return PropertyAddressAllocator::reserved_size(PROPERTY_ADDRESS_ALLOCATOR_ARGS);
        }

        IPropertyPool(asset::SBufferRange<IGPUBuffer>&& _memoryBlock, uint32_t capacity, void* reserved)
            :   memoryBlock(std::move(_memoryBlock)), indexAllocator(reserved,0u,0u,PROPERTY_ADDRESS_ALLOCATOR_ARGS)
        {
			assert(memoryBlock.size>capacity*sizeof(uint32_t)); // this is really a lower bound
        }
        #undef PROPERTY_ADDRESS_ALLOCATOR_ARGS

		virtual ~IPropertyPool() {}


        asset::SBufferRange<IGPUBuffer> memoryBlock;
        PropertyAddressAllocator indexAllocator;
};


}
}

#endif

/*
Old Code

class IMeshSceneNodeInstanced : public ISceneNode
{
    struct MeshLoD
    {
        video::IGPUMesh* mesh;
        void* userDataForVAOSetup; //put array of vertex attribute mappings here or something
        float lodDistance;
    };

    virtual bool setLoDMeshes(const core::vector<MeshLoD>& levelsOfDetail, const size_t& dataSizePerInstanceOutput, const video::SGPUMaterial& lodSelectionShader, VaoSetupOverrideFunc vaoSetupOverride,
        const size_t shaderLoDsPerPass = 1, void* overrideUserData = NULL, const size_t& extraDataSizePerInstanceInput = 0) = 0;

    virtual video::CGPUMesh* getLoDMesh(const size_t& lod) = 0;


    virtual const core::aabbox3df& getLoDInvariantBBox() const = 0;


    inline void setBBoxUpdateEnabled() { wantBBoxUpdate = true; }
    inline void setBBoxUpdateDisabled() { wantBBoxUpdate = false; }
    inline const bool& getBBoxUpdateMode() { return wantBBoxUpdate; }
};
*/