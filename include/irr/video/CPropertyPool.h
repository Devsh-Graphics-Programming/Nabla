// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_H_INCLUDED__

#include "irr/video/IPropertyPool.h"


namespace irr
{
namespace video
{

    
template<template<class> class allocator=core::allocator, typename... Properties>
class CPropertyPool : public IPropertyPool
{
        using ThisType = CPropertyPool<allocator,Properties...>;

        static auto propertyCombinedSize()
        {
            return (sizeof(Properties) + ...);
        }
        static size_t calcApproximateCapacity(size_t bufferSize)
        {
            return bufferSize/propertyCombinedSize();
        }

        _NBL_STATIC_INLINE_CONSTEXPR auto PropertyCount = sizeof...(Properties);

	public:
        static inline core::smart_refctd_ptr<ThisType> create(IVideoDriver* _driver, asset::SBufferRange<IGPUBuffer>&& _memoryBlock, allocator<uint8_t>&& alloc=allocator<uint8_t>())
        {
            if (!_memoryBlock.isValid())
                return nullptr;

            const auto approximateCapacity = calcApproximateCapacity(_memoryBlock.size);
            auto capacity = approximateCapacity;
            while (capacity)
            {
                size_t wouldBeSize = PropertySizes[0]*capacity;
                // now compute with padding and alignments
                for (auto i=1; i<PropertyCount; i++)
                {
                    // align
                    wouldBeSize = core::roundUp(wouldBeSize,PropertySizes[i]);
                    // increase
                    wouldBeSize += PropertySizes[i]*capacity;
                }
                // if still manage to fit, then ok
                if (wouldBeSize<=_memoryBlock.size)
                    break;
                capacity--;
            }
            capacity = core::min<decltype(capacity)>(~uint32_t(0), capacity);
            if (!capacity)
                return nullptr;

            //
            auto reserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc,getReservedSize(capacity));
            if (!reserved)
                return nullptr;

            return core::make_smart_refctd_ptr<CPropertyPool>(_driver,std::move(_memoryBlock),std::move(alloc),capacity,reserved);
        }


        //
        inline uint32_t getPipelineCount() const override
        {
            return core::roundUp(PropertyCount,maxPipelinesPerPass);
        }

        //
        inline void getPipelines(core::smart_refctd_ptr<IGPUComputePipeline>* outIt, bool forDownload, bool canCompileNew, IGPUPipelineCache* pipelineCache=nullptr) const override
        {
            PipelineKey key = {forDownload,{}};
            for (uint32_t i=0u; i<getPipelineCount(); i++)
            {
                auto propertiesThisPass = getPropertiesPerPass(i);
                // need a redirect/shuffle cause need sorted elements
                for (uint32_t j=0u; j<propertiesThisPass; j++)
                    key.propertySizes[j] = getShuffledGlobalIndex(i,j);
                // no uninit vars
                for (uint32_t j=propertiesThisPass; j<key.propertySizes.size(); j++)
                    key.propertySizes[j] = 0u;
                //
                *(outIt++) = IPropertyPool::getCopyPipeline(driver,key,canCompileNew,pipelineCache);
            }
        }

	protected:
        inline uint32_t getPropertiesPerPass(uint32_t passID)
        {
            if (passID!=(getPipelineCount()-1u))
                return maxPipelinesPerPass;

            return PropertyCount-passID*maxPipelinesPerPass;
        }
        inline uint32_t getUnshuffledGlobalIndex(uint32_t passID, uint32_t localID)
        {
            return passID*maxPipelinesPerPass+localID;
        }
        inline auto& getShuffledGlobalIndex(uint32_t passID, uint32_t localID)
        {
            return copyPipelinePropertyRedirect[getUnshuffledGlobalIndex(i,j)];
        }

        CPropertyPool(IVideoDriver* _driver, core::SBufferRange<IGPUBuffer>&& _memoryBlock, allocator<uint8_t>&& _alloc, uint32_t capacity, void* reserved)
            : IPropertyPool(_driver,std::move(_memoryBlock),capacity,reserved), maxPipelinesPerPass((16/*driver->getMaxSSBOBindings(ESS_COMPUTE)*/-1)/2u), alloc(std::move(_alloc))
        {
            // fill out redirects
            std::pair<uint32_t,uint32_t> tmp[maxPipelinesPerPass];
            for (uint32_t i=0u; i<getPipelineCount(); i++)
            {
                auto propertiesThisPass = getPropertiesPerPass(i);
                //
                for (uint32_t j=0u; j<propertiesThisPass; j++)
                    tmp[j] = {PropertySizes[getUnshuffledGlobalIndex(i,j)],getUnshuffledGlobalIndex(i,j)};
                std::sort(tmp,tmp+propertiesThisPass);
                // need a redirect/shuffle cause need sorted elements
                for (uint32_t j=0u; j<propertiesThisPass; j++)
                    getShuffledGlobalIndex(i,j) = tmp[j].second;
            }
        }

        ~CPropertyPool()
        {
            void* reserved = const_cast<void*>(indexAllocator.getReservedSpacePtr());
            std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,reserved,getReservedSize(getCapacity()));
        }

        
        uint32_t maxPipelinesPerPass;
        // the size is just an upper bound
        std::array<uint32_t,PropertyCount> copyPipelinePropertyRedirect = {};
        //
        allocator<uint8_t> alloc;

        _NBL_STATIC_INLINE_CONSTEXPR std::array<uint32_t,PropertyCount+1u> PropertySizes = {sizeof(Properties)... , 0u};
};


}
}

#endif