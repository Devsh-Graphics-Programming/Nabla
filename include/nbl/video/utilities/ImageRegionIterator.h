// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_IMAGE_REGION_ITERATOR_H_INCLUDED_
#define _NBL_VIDEO_IMAGE_REGION_ITERATOR_H_INCLUDED_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"

namespace nbl::video
{

class ImageRegionIterator
{
    public:
        ImageRegionIterator(
            const std::span<const asset::IImage::SBufferCopy> copyRegions,
            IPhysicalDevice::SQueueFamilyProperties queueFamilyProps,
            const void* srcData,
            asset::E_FORMAT srcImageFormat,
            video::IGPUImage* const dstImage,
            size_t optimalRowPitchAlignment
        );
    
        // ! Memory you need to allocate to transfer the remaining regions in one submit.
        // ! WARN: It's okay to use less memory than the return value of this function for your staging memory, in that usual case more than 1 copy regions will be needed to transfer the remaining regions.
        size_t getMemoryNeededForRemainingRegions() const;

        // ! Gives `regionToCopyNext` based on `availableMemory`
        // ! memcopies the data from `srcData` to `stagingBuffer`, preparing it for launch and submit to copy to GPU buffer
        // ! updates `availableMemory` (availableMemory -= consumedMemory)
        // ! updates `stagingBufferOffset` based on consumed memory and alignment requirements
        // ! this function may do format conversions when copying from `srcData` to `stagingBuffer` if srcImageFormat != dstImage->getCreationParams().format passed as constructor parameters
        bool advanceAndCopyToStagingBuffer(asset::IImage::SBufferCopy& regionToCopyNext, uint32_t& availableMemory, uint32_t& stagingBufferOffset, void* stagingBufferPointer);

        // ! returns true when there is no more regions left over to copy
        bool isFinished() const { return currentRegion == regions.size(); }
        uint32_t getCurrentBlockInRow() const { return currentBlockInRow; }
        uint32_t getCurrentRowInSlice() const { return currentRowInSlice; }
        uint32_t getCurrentSliceInLayer() const { return currentSliceInLayer; }
        uint32_t getCurrentLayerInRegion() const { return currentLayerInRegion; }
        uint32_t getCurrentRegion() const { return currentRegion; }

        inline core::vector3du32_SIMD getOptimalCopyTexelStrides(const asset::VkExtent3D& copyExtents) const
        {
            return core::vector3du32_SIMD(
                core::alignUp(copyExtents.width, optimalRowPitchAlignment),
                copyExtents.height,
                copyExtents.depth);
        }

    private:
        const std::span<const asset::IImage::SBufferCopy> regions;

        // Mock CPU Images used to copy cpu buffer to staging buffer
        // TODO: unless you're doing region copies in parallel, just have one image each and set new regions each upload
        std::vector<core::smart_refctd_ptr<asset::ICPUImage>> imageFilterInCPUImages;
        core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy> outCPUImageRegions; // needs to be updated before each upload
        std::vector<core::smart_refctd_ptr<asset::ICPUImage>> imageFilterOutCPUImages;

        size_t optimalRowPitchAlignment = 1u;
        bool canTransferMipLevelsPartially = false;
        const asset::VkExtent3D minImageTransferGranularity = {};
        uint32_t bufferOffsetAlignment = 1u;

        asset::E_FORMAT srcImageFormat;
        const asset::E_FORMAT dstImageFormat;
        const void* const srcData;
        video::IGPUImage* const dstImage;
    
        // Block Offsets 
        uint16_t currentBlockInRow = 0u;
        uint16_t currentRowInSlice = 0u;
        uint16_t currentSliceInLayer = 0u;
        uint16_t currentLayerInRegion = 0u;
        uint16_t currentRegion = 0u;
};

}
#endif