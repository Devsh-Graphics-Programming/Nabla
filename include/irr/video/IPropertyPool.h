#ifndef __IRR_VIDEO_I_PROPERTY_POOL_H_INCLUDED__
#define __IRR_VIDEO_I_PROPERTY_POOL_H_INCLUDED__


#include "irr/asset/asset.h"

#include "IVideoDriver.h"
#include "irr/video/IGPUComputePipeline.h"


namespace irr
{
namespace video
{


class IPropertyPool : public core::IReferenceCounted
{
	public:
		using PropertyAddressAllocator = core::PoolAddressAllocatorST<uint32_t>;

        _IRR_STATIC_INLINE_CONSTEXPR auto InvalidIndex = PropertyAddressAllocator::invalid_address;


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


        _IRR_STATIC_INLINE_CONSTEXPR auto MaxPropertiesPerCS = 15;
        struct PipelineKey
        {
            uint32_t getPropertyCount() const
            {
                if (!PropertySizes[0])
                    return 0u;

                for (auto i=1; i<MaxPropertiesPerCS; i++)
                {
                    if (PropertySizes[i]>PropertySizes[i-1])
                        return 0u;
                    if (!PropertySizes[i])
                        return i;
                }

                return MaxPropertiesPerCS;
            }

            bool download;
            std::array<uint32_t,MaxPropertiesPerCS> propertySizes;
        };
        //
        static core::smart_refctd_ptr<IGPUComputePipeline> getCopyPipeline(IVideoDriver* driver, const PipelineKey& key, bool canCompileNew, IGPUPipelineCache* pipelineCache=nullptr);

        //
        virtual uint32_t getPipelineCount() const = 0;
        //
        virtual void getPipelines(core::smart_refctd_ptr<IGPUComputePipeline>* outIt, bool forDownload, bool canCompileNew, IGPUPipelineCache* pipelineCache=nullptr) const = 0;


        // allocate
        inline bool allocateProperties(uint32_t* outIndicesBegin, uint32_t* outIndicesEnd)
        {
            constexpr uint32_t unit = 1u;
            for (auto it=outIndicesBegin; it!=outIndicesEnd; it++)
            {
                auto& addr = it;
                if (addr!=InvalidIndex)
                    continue;

                addr = indexAllocator.alloc_addr(unit,unit);
                if (addr==InvalidIndex)
                    return false;
            }
            return true;
        }

        //
        inline core::smart_refctd_ptr<IDriverFence> uploadProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd, std::initializer_list<const void*> data)
        {
            return nullptr;
        }

        //
        inline core::smart_refctd_ptr<IDriverFence> addProperties(uint32_t* outIndicesBegin, uint32_t* outIndicesEnd, std::initializer_list<const void*> data)
        {
            if (!allocateProperties(outIndicesBegin,outIndicesEnd))
                return nullptr;

            return uploadProperties(outIndicesBegin,outIndicesEnd,data);
        }

        //
        virtual core::smart_refctd_ptr<IDriverFence> downloadProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd, std::initializer_list<void*> data)
        {
            const auto passCount = getPipelineCount();
            core::smart_refctd_ptr<IGPUComputePipeline> passes[passCount];
            for (auto pass=0u; pass<passCount; pass++)
            {
                //driver->bindComputePipeline();
                //driver->bindDescriptorSets(EPBP_COMPUTE,layout,0u,1u,&set,&offsets);
                //driver->dispatch(getWorkGroupSizeX(propertiesThisPass),propertiesThisPass,1u);
            }
            return nullptr;
        }

        //
        inline void freeProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd)
        {
            constexpr uint32_t unit = 1u;
            for (auto it=outIndicesBegin; it!=outIndicesEnd; it++)
            {
                auto& addr = it;
                if (addr!=InvalidIndex)
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
        static inline getReservedSize(uint32_t capacity)
        {
            return PropertyAddressAllocator::reserved_size(PROPERTY_ADDRESS_ALLOCATOR_ARGS);
        }

        IPropertyPool(IVideoDriver* _driver, asset::SBufferRange<IGPUBuffer>&& _memoryBlock, uint32_t capacity, void* reserved)
            :   driver(_driver), memoryBlock(std::move(_memoryBlock)), indexAllocator(IPropertyPool(std::move(_memoryBlock),
                    PropertyAddressAllocator(reserved,0u,0u,PROPERTY_ADDRESS_ALLOCATOR_ARGS)
                )
        {
        }
        #undef PROPERTY_ADDRESS_ALLOCATOR_ARGS

        virtual ~IPropertyPool() {}

        //
        constexpr auto IdealWorkGroupSize = 256u;
        static inline uint32_t getWorkGroupSizeX(uint32_t propertyCount)
        {
            return core::roundDownToPoT(IdealWorkGroupSize/propertyCount);
        }


        IVideoDriver* driver;
        asset::SBufferRange<IGPUBuffer>&& memoryBlock;
        PropertyAddressAllocator indexAllocator;

        // waiting for @Hazard
        template<typename Key, typename Value>
        using LRUCache = core::unordered_map<Key,Value>;

        static LRUCache<PipelineKey,core::smart_refctd_ptr<IGPUComputePipeline>> copyPipelines;
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