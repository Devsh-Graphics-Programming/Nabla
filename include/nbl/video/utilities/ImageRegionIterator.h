// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_IMAGE_REGION_ITERATOR_H_INCLUDED_
#define _NBL_VIDEO_IMAGE_REGION_ITERATOR_H_INCLUDED_

#include "nbl/asset/asset.h"

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"

namespace nbl::video
{

// ! Chops `copyRegions` of an image into chunks that fit a staging buffer allocation.
// ! Direction agnostic: a chunk is what one `copyBufferToImage` or `copyImageToBuffer` region touches, and enough to also copy the
// ! chunk between the staging buffer and the host memory `copyRegions` describe, see `copyChunk`.
// ! The chunking respects both copy commands' shared valid usage on `bufferOffset` alignment and transfer granularity:
// ! VUID-vkCmdCopyBufferToImage-dstImage-07975 / VUID-vkCmdCopyImageToBuffer-srcImage-07975 (texel block size)
// ! VUID-vkCmdCopyBufferToImage-dstImage-07978 / VUID-vkCmdCopyImageToBuffer-srcImage-07978 (depth/stencil, multiple of 4)
// ! VUID-vkCmdCopyBufferToImage-commandBuffer-07737 / VUID-vkCmdCopyImageToBuffer-commandBuffer-07746 (no graphics/compute queue, multiple of 4)
// ! VUID-vkCmdCopyBufferToImage-imageOffset-07738 / VUID-vkCmdCopyImageToBuffer-imageOffset-07747 (minImageTransferGranularity)
class ImageRegionIterator
{
    public:
        struct SChunk
        {
            // ! Region for the `copyBufferToImage`/`copyImageToBuffer` call, `bufferOffset` is relative to the staging buffer
            asset::IImage::SBufferCopy staging = {};
            // ! Bytes the staging region spans, the rows are laid out at `getOptimalCopyTexelStrides` pitch
            uint32_t stagingByteSize = 0u;
            // ! Format of the image, and hence of the staged data
            asset::E_FORMAT imageFormat = asset::EF_UNKNOWN;
            // ! `bufferOffset` of the region in `copyRegions` this chunk was cut from
            size_t hostBufferOffset = 0ull;
            // ! `SBufferCopy::getTexelStrides()` of the region in `copyRegions` this chunk was cut from
            core::vector3du32_SIMD hostTexelStrides = core::vector3du32_SIMD(0u,0u,0u);
            // ! Where the chunk starts inside the region it was cut from, xyz in texels and w the array layer
            core::vector4du32_SIMD hostOffsetBaseLayer = core::vector4du32_SIMD(0u,0u,0u,0u);
        };

        // ! The extra copy between the host memory `copyRegions` describe and the staging buffer.
        enum class E_DIRECTION : uint8_t
        {
            // ! host -> staging, before `copyBufferToImage`
            HOST_TO_STAGING,
            // ! staging -> host, after `copyImageToBuffer` completed
            STAGING_TO_HOST
        };

        ImageRegionIterator(
            const std::span<const asset::IImage::SBufferCopy> copyRegions,
            IPhysicalDevice::SQueueFamilyProperties queueFamilyProps,
            const video::IGPUImage* const image,
            size_t optimalRowPitchAlignment
        );

        // ! Memory you need to allocate to transfer the remaining regions in one submit.
        // ! WARN: It's okay to use less memory than the return value of this function for your staging memory, in that usual case more than 1 copy regions will be needed to transfer the remaining regions.
        size_t getMemoryNeededForRemainingRegions() const;

        // ! Cuts the next `chunk` that fits in `availableMemory`, largest transferable unit first (layers, slices, rows, blocks)
        // ! updates `availableMemory` (availableMemory -= consumedMemory)
        // ! updates `stagingBufferOffset` based on consumed memory and alignment requirements
        // ! does not touch any memory, @see `copyChunk`
        bool advance(SChunk& chunk, uint32_t& availableMemory, uint32_t& stagingBufferOffset);

        // ! Copies one chunk between `hostPtr` (laid out as the region the chunk was cut from, in `hostFormat`) and `stagingPtr` (the staging buffer base).
        // ! Same format: `CBasicImageFilterCommon::copyBlocks`, memcpy of the largest runs of rows, slices or layers without padding on either side, padding is never read or written.
        // ! Different formats: maps channels by position, missing ones to zero and alpha to one, with `STexelReshuffle` when no channel changes its encoding, otherwise with `CSwizzleAndConvertImageFilter` on mock `ICPUImage`s.
        static bool copyChunk(const SChunk& chunk, const E_DIRECTION direction, void* stagingPtr, void* hostPtr, const asset::E_FORMAT hostFormat);

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

        size_t optimalRowPitchAlignment = 1u;
        bool canTransferMipLevelsPartially = false;
        const asset::VkExtent3D minImageTransferGranularity = {};
        uint32_t bufferOffsetAlignment = 1u;

        const asset::E_FORMAT imageFormat;
        const video::IGPUImage* const image;

        // Block Offsets
        uint32_t currentBlockInRow = 0u;
        uint32_t currentRowInSlice = 0u;
        uint32_t currentSliceInLayer = 0u;
        uint32_t currentLayerInRegion = 0u;
        uint32_t currentRegion = 0u;
};

}
#endif
