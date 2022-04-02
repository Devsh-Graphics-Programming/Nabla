#include "nbl/video/utilities/IUtilities.h"
#include <numeric>

namespace nbl::video
{

void IUtilities::updateImageViaStagingBuffer(
    IGPUCommandBuffer* cmdbuf, IGPUFence* fence, IGPUQueue* queue,
    asset::ICPUBuffer const* srcBuffer, const core::SRange<const asset::IImage::SBufferCopy>& regions, video::IGPUImage* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout,
    uint32_t& waitSemaphoreCount, IGPUSemaphore*const * &semaphoresToWaitBeforeOverwrite, const asset::E_PIPELINE_STAGE_FLAGS* &stagesToWaitForPerSemaphore)
{
    const auto& limits = m_device->getPhysicalDevice()->getLimits();
    const uint32_t allocationAlignment = static_cast<uint32_t>(limits.nonCoherentAtomSize);

    auto* cmdpool = cmdbuf->getPool();
    assert(cmdbuf->isResettable());
    assert(cmdpool->getQueueFamilyIndex()==queue->getFamilyIndex());
            
    auto texelBlockInfo = asset::TexelBlockInfo(dstImage->getCreationParameters().format);
    auto texelBlockDim = texelBlockInfo.getDimension();
    auto queueFamProps = m_device->getPhysicalDevice()->getQueueFamilyProperties()[0];
    auto minImageTransferGranularity = queueFamProps.minImageTransferGranularity;

    // Queues supporting graphics and/or compute operations must report (1,1,1) in minImageTransferGranularity, meaning that there are no additional restrictions on the granularity of image transfer operations for these queues.
    // Other queues supporting image transfer operations are only required to support whole mip level transfers, thus minImageTransferGranularity for queues belonging to such queue families may be (0,0,0)
    bool canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);

    // Block Offsets 
    // (1 blockInRow = texelBlockDimensions.x texels)
    // (1 rowInSlice = texelBlockDimensions.y texel rows)
    // (1 sliceInLayer = texelBlockDimensions.z texel depths)
    uint32_t currentBlockInRow = 0u;
    uint32_t currentRowInSlice = 0u;
    uint32_t currentSliceInLayer = 0u;
    uint32_t currentLayerInRegion = 0u;
    uint32_t currentRegion = 0u;
            
    // bufferOffsetAlignment:
        // [x] If Depth/Stencil -> must be multiple of 4
        // [ ] If multi-planar -> bufferOffset must be a multiple of the element size of the compatible format for the aspectMask of imagesubresource
        // [x] If Queue doesn't support GRAPHICS_BIT or COMPUTE_BIT ->  must be multiple of 4
        // [x] bufferOffset must be a multiple of texel block size in bytes
    uint32_t bufferOffsetAlignment = texelBlockInfo.getBlockByteSize();
    if(asset::isDepthOrStencilFormat(dstImage->getCreationParameters().format))
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);

    bool queueSupportsCompute = (queueFamProps.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
    bool queueSupportsGraphics = (queueFamProps.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
    if((queueSupportsGraphics || queueSupportsCompute) == false)
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);

    // TODO: Need to have a function to get equivalent format of the specific plane of this format (in aspectMask)
    // if(asset::isPlanarFormat(dstImage->getCreationParameters().format))

    assert(core::is_alignment(bufferOffsetAlignment));
    
    // Assuming each thread can handle minImageTranferGranularitySize of texelBlocks:
    const uint32_t maxResidentImageTransferSize = limits.maxResidentInvocations * texelBlockInfo.getBlockByteSize() * (minImageTransferGranularity.width * minImageTransferGranularity.height * minImageTransferGranularity.depth); 
    // memoryLowerBound = max(maxResidentImageTransferSize, the largest rowPitch of regions); 
    uint32_t memoryLowerBound = maxResidentImageTransferSize;

    while (currentRegion < regions.size())
    {
        size_t memoryNeededForRemainingRegions = 0ull;
        for (uint32_t i = currentRegion; i < regions.size(); ++i)
        {
            memoryNeededForRemainingRegions = core::alignUp(memoryNeededForRemainingRegions, bufferOffsetAlignment);

            const asset::IImage::SBufferCopy & region = regions[i];

            auto subresourceSize = dstImage->getMipSize(region.imageSubresource.mipLevel);

            assert(static_cast<uint32_t>(region.imageSubresource.aspectMask) != 0u);
            assert(core::isPoT(static_cast<uint32_t>(region.imageSubresource.aspectMask)) && "region.aspectMask should only have a single bit set.");
            // Validate Region
            // canTransferMipLevelsPartially
            if(!canTransferMipLevelsPartially)
            {
                assert(region.imageOffset.x == 0 && region.imageOffset.y == 0 && region.imageOffset.z == 0);
                assert(region.imageExtent.width == subresourceSize.x && region.imageExtent.height == subresourceSize.y && region.imageExtent.depth == subresourceSize.z);
            }

            // region.imageOffset.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size
            bool isImageOffsetAlignmentValid =
                (region.imageOffset.x % (minImageTransferGranularity.width  * texelBlockDim.x) == 0) &&
                (region.imageOffset.y % (minImageTransferGranularity.height * texelBlockDim.y) == 0) &&
                (region.imageOffset.z % (minImageTransferGranularity.depth  * texelBlockDim.z) == 0);
            assert(isImageOffsetAlignmentValid);

            // region.imageExtent.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size,
            // OR ELSE (region.imageOffset.{x/y/z} + region.imageExtent.{width/height/depth}) MUST be equal to subresource{Width,Height,Depth}
            bool isImageExtentAlignmentValid = 
                (region.imageExtent.width  % (minImageTransferGranularity.width  * texelBlockDim.x) == 0 || (region.imageOffset.x + region.imageExtent.width   == subresourceSize.x)) && 
                (region.imageExtent.height % (minImageTransferGranularity.height * texelBlockDim.y) == 0 || (region.imageOffset.y + region.imageExtent.height  == subresourceSize.y)) &&
                (region.imageExtent.depth  % (minImageTransferGranularity.depth  * texelBlockDim.z) == 0 || (region.imageOffset.z + region.imageExtent.depth   == subresourceSize.z));
            assert(isImageExtentAlignmentValid);

            bool isImageExtentAndOffsetValid = 
                (region.imageExtent.width + region.imageOffset.x <= subresourceSize.x) &&
                (region.imageExtent.height + region.imageOffset.y <= subresourceSize.y) &&
                (region.imageExtent.depth + region.imageOffset.z <= subresourceSize.z);
            assert(isImageExtentAndOffsetValid);

            auto imageExtent = core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth);
            auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(imageExtent);
            auto imageExtentBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);
            memoryLowerBound = core::max(memoryLowerBound, imageExtentBlockStridesInBytes[1]); // rowPitch = imageExtentBlockStridesInBytes[1]

            if(i == currentRegion)
            {
                auto remainingBlocksInRow = imageExtentInBlocks.x - currentBlockInRow;
                auto remainingRowsInSlice = imageExtentInBlocks.y - currentRowInSlice;
                auto remainingSlicesInLayer = imageExtentInBlocks.z - currentSliceInLayer;
                auto remainingLayersInRegion = region.imageSubresource.layerCount - currentLayerInRegion;

                if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && remainingLayersInRegion > 0)
                    memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * remainingLayersInRegion;
                else if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer > 0)
                {
                    memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[2] * remainingSlicesInLayer;
                    if(remainingLayersInRegion > 1u)
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u);
                }
                else if (currentBlockInRow == 0 && currentRowInSlice > 0)
                {
                    memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[1] * remainingRowsInSlice;
                    if(remainingSlicesInLayer > 1u)
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[2] * (remainingSlicesInLayer - 1u);
                    if(remainingLayersInRegion > 1u)
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u);
                }
                else if (currentBlockInRow > 0)
                {
                    memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[0] * remainingBlocksInRow;
                    if(remainingRowsInSlice > 1u)
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[1] * (remainingRowsInSlice - 1u);
                    if(remainingSlicesInLayer > 1u)
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[2] * (remainingSlicesInLayer - 1u);
                    if(remainingLayersInRegion > 1u)
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u);
                }
            }
            else
            {
                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * region.imageSubresource.layerCount; // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z * region.imageSubresource.layerCount
            }
        }

        uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t subSize = static_cast<uint32_t>(core::min<uint64_t>(core::alignDown(m_defaultUploadBuffer.get()->max_size(), allocationAlignment), memoryNeededForRemainingRegions));
        subSize = std::max(subSize, memoryLowerBound);
        const uint32_t uploadBufferSize = core::alignUp(subSize, allocationAlignment);
        // cannot use `multi_place` because of the extra padding size we could have added
        m_defaultUploadBuffer.get()->multi_alloc(std::chrono::steady_clock::now()+std::chrono::microseconds(500u), 1u, &localOffset, &uploadBufferSize, &allocationAlignment);
        bool failedAllocation = (localOffset == video::StreamingTransientDataBufferMT<>::invalid_address);

        // keep trying again
        if (failedAllocation)
        {
            if(currentRegion == 0 && currentLayerInRegion == 0 && currentSliceInLayer == 0 && currentRowInSlice == 0 && currentBlockInRow == 0)
            {
                // TODO: Log Failed Allocation and return false;
                _NBL_DEBUG_BREAK_IF(false && "Failed Initial Allocation.");
                break;
            }

            // but first sumbit the already buffered up copies
            cmdbuf->end();
            IGPUQueue::SSubmitInfo submit;
            submit.commandBufferCount = 1u;
            submit.commandBuffers = &cmdbuf;
            submit.signalSemaphoreCount = 0u;
            submit.pSignalSemaphores = nullptr;
            assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
            submit.waitSemaphoreCount = waitSemaphoreCount;
            submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
            submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
            queue->submit(1u, &submit, fence);
            m_device->blockForFences(1u, &fence);
            waitSemaphoreCount = 0u;
            semaphoresToWaitBeforeOverwrite = nullptr;
            stagesToWaitForPerSemaphore = nullptr;
            // before resetting we need poll all events in the allocator's deferred free list
            m_defaultUploadBuffer->cull_frees();
            // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
            m_device->resetFences(1u, &fence);
            cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
            continue;
        }
        else
        {
            // Start CmdCopying Regions and Copying Data to m_defaultUploadBuffer
            uint32_t currentUploadBufferOffset = 0u;
            uint32_t availableUploadBufferMemory = 0u;
            auto addToCurrentUploadBufferOffset = [&](uint32_t size) -> bool 
            {
                currentUploadBufferOffset += size;
                currentUploadBufferOffset = core::alignUp(currentUploadBufferOffset, bufferOffsetAlignment);
                if(currentUploadBufferOffset - localOffset <= uploadBufferSize)
                {
                    availableUploadBufferMemory = uploadBufferSize - (currentUploadBufferOffset - localOffset);
                    return true;
                }
                else
                    return false;
            };

            // currentUploadBufferOffset = localOffset
            // currentUploadBufferOffset = alignUp(currentUploadBufferOffset, bufferOffsetAlignment)
            // availableUploadBufferMemory = uploadBufferSize
            addToCurrentUploadBufferOffset(localOffset);

            bool anyTransferRecorded = false;
            core::vector<asset::IImage::SBufferCopy> regionsToCopy;

            // Worst case iterations: remaining blocks --> remaining rows --> remaining slices --> full layers
            const uint32_t maxIterations = regions.size() * 4u; 
            for (uint32_t d = 0u; d < maxIterations && currentRegion < regions.size(); ++d)
            {
                const asset::IImage::SBufferCopy & region = regions[currentRegion];

                auto subresourceSize = dstImage->getMipSize(region.imageSubresource.mipLevel);
                auto subresourceSizeInBlocks = texelBlockInfo.convertTexelsToBlocks(subresourceSize);

                // regionBlockStrides = <BufferRowLengthInBlocks, BufferImageHeightInBlocks, ImageDepthInBlocks>
                auto regionBlockStrides = region.getBlockStrides(texelBlockInfo);
                // regionBlockStridesInBytes = <BlockByteSize,
                //                              BlockBytesSize * BufferRowLengthInBlocks,
                //                              BlockBytesSize * BufferRowLengthInBlocks * BufferImageHeightInBlocks,
                //                              BlockBytesSize * BufferRowLengthInBlocks * BufferImageHeightInBlocks * ImageDepthInBlocks>
                auto regionBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(regionBlockStrides);

                auto imageExtent = core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth);
                auto imageOffset = core::vector3du32_SIMD(region.imageOffset.x, region.imageOffset.y, region.imageOffset.z);
                auto imageOffsetInBlocks = texelBlockInfo.convertTexelsToBlocks(imageOffset);
                auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(imageExtent);
                auto imageExtentBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);

                // region <-> region.imageSubresource.layerCount <-> imageExtentInBlocks.z <-> imageExtentInBlocks.y <-> imageExtentInBlocks.x
                auto updateCurrentOffsets = [&]() -> void
                {
                    if(currentBlockInRow >= imageExtentInBlocks.x) 
                    {
                        currentBlockInRow = 0u;
                        currentRowInSlice++;
                    }
                    if(currentRowInSlice >= imageExtentInBlocks.y)
                    {
                        assert(currentBlockInRow == 0);
                        currentRowInSlice = 0u;
                        currentSliceInLayer++;
                    }
                    if(currentSliceInLayer >= imageExtentInBlocks.z)
                    {
                        assert(currentBlockInRow == 0 && currentRowInSlice == 0);
                        currentSliceInLayer = 0u;
                        currentLayerInRegion++;
                    }
                    if(currentLayerInRegion >= region.imageSubresource.layerCount) 
                    {
                        assert(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0);
                        currentLayerInRegion = 0u; 
                        currentRegion++;
                    }
                };
                        
                uint32_t eachBlockNeededMemory  = imageExtentBlockStridesInBytes[0];  // = blockByteSize
                uint32_t eachRowNeededMemory    = imageExtentBlockStridesInBytes[1];  // = blockByteSize * imageExtentInBlocks.x
                uint32_t eachSliceNeededMemory  = imageExtentBlockStridesInBytes[2];  // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y
                uint32_t eachLayerNeededMemory  = imageExtentBlockStridesInBytes[3];  // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z

                auto tryFillRow = [&]() -> bool
                {
                    bool ret = false;
                    // C: There is remaining slices left in layer -> Copy Blocks
                    uint32_t uploadableBlocks = availableUploadBufferMemory / eachBlockNeededMemory;
                    uint32_t remainingBlocks = imageExtentInBlocks.x - currentBlockInRow;
                    uploadableBlocks = core::min(uploadableBlocks, remainingBlocks);
                    if(uploadableBlocks > 0 && minImageTransferGranularity.width > 1u && (imageOffsetInBlocks.x + currentBlockInRow + uploadableBlocks) < subresourceSizeInBlocks.x)
                        uploadableBlocks = core::alignDown(uploadableBlocks, minImageTransferGranularity.width);

                    if(uploadableBlocks > 0)
                    {
                        uint32_t blocksToUploadMemorySize = eachBlockNeededMemory * uploadableBlocks;
                        auto localImageOffset = core::vector3du32_SIMD(0u + currentBlockInRow, 0u + currentRowInSlice, 0u + currentSliceInLayer, currentLayerInRegion);
                        uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                        uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                        memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                blocksToUploadMemorySize);

                        asset::IImage::SBufferCopy bufferCopy;
                        bufferCopy.bufferOffset = currentUploadBufferOffset;
                        bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                        bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                        bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                        bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                        bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                        bufferCopy.imageOffset.x = region.imageOffset.x + currentBlockInRow * texelBlockDim.x;
                        bufferCopy.imageOffset.y = region.imageOffset.y + currentRowInSlice * texelBlockDim.y;
                        bufferCopy.imageOffset.z = region.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
                        bufferCopy.imageExtent.width    = core::min(uploadableBlocks * texelBlockDim.x, imageExtent.x);
                        bufferCopy.imageExtent.height   = core::min(1u * texelBlockDim.y, imageExtent.y);
                        bufferCopy.imageExtent.depth    = core::min(1u * texelBlockDim.z, imageExtent.z);
                        bufferCopy.imageSubresource.layerCount = 1u;
                        regionsToCopy.push_back(bufferCopy);

                        addToCurrentUploadBufferOffset(blocksToUploadMemorySize);

                        currentBlockInRow += uploadableBlocks;
                        ret = true;
                    }

                    updateCurrentOffsets();

                    return ret;
                };
                        
                auto tryFillSlice = [&]() -> bool
                {
                    bool ret = false;
                    // B: There is remaining slices left in layer -> Copy Rows
                    uint32_t uploadableRows = availableUploadBufferMemory / eachRowNeededMemory;
                    uint32_t remainingRows = imageExtentInBlocks.y - currentRowInSlice;
                    uploadableRows = core::min(uploadableRows, remainingRows);
                    if(uploadableRows > 0 && minImageTransferGranularity.height > 1u && (imageOffsetInBlocks.y + currentRowInSlice + uploadableRows) < subresourceSizeInBlocks.y)
                        uploadableRows = core::alignDown(uploadableRows, minImageTransferGranularity.height);

                    if(uploadableRows > 0)
                    {
                        uint32_t rowsToUploadMemorySize = eachRowNeededMemory * uploadableRows;
                                
                        if(regionBlockStrides.x != imageExtentInBlocks.x)
                        {
                            // Can't copy all rows at once, there is padding, copy row by row
                            for(uint32_t y = 0; y < uploadableRows; ++y)
                            {
                                auto localImageOffset = core::vector3du32_SIMD(0u, 0u + currentRowInSlice + y, 0u + currentSliceInLayer, currentLayerInRegion);
                                uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                                uint64_t offsetInUploadBuffer = currentUploadBufferOffset + y*eachRowNeededMemory;
                                memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                        reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                        eachRowNeededMemory);
                            }
                        }
                        else
                        {
                            // We can copy all rows at once, because imageExtent is fit to rowLength
                            assert(imageOffsetInBlocks.x == 0);
                            auto localImageOffset = core::vector3du32_SIMD(0u, 0u + currentRowInSlice, 0u + currentSliceInLayer, currentLayerInRegion);
                            uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                            uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                            memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                    reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                    rowsToUploadMemorySize);
                        }


                        asset::IImage::SBufferCopy bufferCopy;
                        bufferCopy.bufferOffset = currentUploadBufferOffset;
                        bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                        bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                        bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                        bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                        bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                        bufferCopy.imageOffset.x = region.imageOffset.x + 0u; assert(currentBlockInRow == 0);
                        bufferCopy.imageOffset.y = region.imageOffset.y + currentRowInSlice * texelBlockDim.y;
                        bufferCopy.imageOffset.z = region.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
                        bufferCopy.imageExtent.width    = imageExtent.x;
                        bufferCopy.imageExtent.height   = core::min(uploadableRows * texelBlockDim.y, imageExtent.y);
                        bufferCopy.imageExtent.depth    = core::min(1u * texelBlockDim.z, imageExtent.z);
                        bufferCopy.imageSubresource.layerCount = 1u;
                        regionsToCopy.push_back(bufferCopy);

                        addToCurrentUploadBufferOffset(rowsToUploadMemorySize);

                        currentRowInSlice += uploadableRows;
                        ret = true;
                    }
                            
                    if(currentRowInSlice < imageExtentInBlocks.y)
                    {
                        bool filledAnyBlocksInRow = tryFillRow();
                        if(filledAnyBlocksInRow)
                            ret = true;
                    }
                            
                    updateCurrentOffsets();

                    return ret;
                };
                        
                auto tryFillLayer = [&]() -> bool
                {
                    bool ret = false;
                    // A: There is remaining layers left in region -> Copy Slices (Depths)
                    uint32_t uploadableSlices = availableUploadBufferMemory / eachSliceNeededMemory;
                    uint32_t remainingSlices = imageExtentInBlocks.z - currentSliceInLayer;
                    uploadableSlices = core::min(uploadableSlices, remainingSlices);
                    if(uploadableSlices > 0 && minImageTransferGranularity.depth > 1u && (imageOffsetInBlocks.z + currentSliceInLayer + uploadableSlices) < subresourceSizeInBlocks.z)
                        uploadableSlices = core::alignDown(uploadableSlices, minImageTransferGranularity.depth);

                    if(uploadableSlices > 0)
                    {
                        uint32_t slicesToUploadMemorySize = eachSliceNeededMemory * uploadableSlices;

                        if(regionBlockStrides.x != imageExtentInBlocks.x)
                        {
                            // Can't copy all rows at once, there is more padding at the end of rows, copy row by row:
                            for(uint32_t z = 0; z < uploadableSlices; ++z)
                            {
                                for(uint32_t y = 0; y < imageExtentInBlocks.y; ++y)
                                {
                                    auto localImageOffset = core::vector3du32_SIMD(0u, 0u + y, 0u + currentSliceInLayer + z, currentLayerInRegion);
                                    uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                                    uint64_t offsetInUploadBuffer = currentUploadBufferOffset + z * eachSliceNeededMemory + y * eachRowNeededMemory;
                                    memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                            reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                            eachRowNeededMemory);
                                }
                            }
                        }
                        else if (regionBlockStrides.y != imageExtentInBlocks.y)
                        {
                            assert(imageOffsetInBlocks.x == 0u);
                            // Can't copy all slices at once, there is more padding at the end of slices, copy slice by slice
                            for(uint32_t z = 0; z < uploadableSlices; ++z)
                            {
                                auto localImageOffset = core::vector3du32_SIMD(0u, 0u, 0u + currentSliceInLayer + z, currentLayerInRegion);
                                uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                                uint64_t offsetInUploadBuffer = currentUploadBufferOffset + z * eachSliceNeededMemory;
                                memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                        reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                        eachSliceNeededMemory);
                            }
                        }
                        else
                        {
                            // We can copy all arrays and slices at once, because imageExtent is fit to bufferRowLength and bufferImageHeight
                            assert(imageOffsetInBlocks.x == 0u);
                            assert(imageOffsetInBlocks.y == 0u);
                            auto localImageOffset = core::vector3du32_SIMD(0u, 0u, 0u + currentSliceInLayer, currentLayerInRegion);
                            uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                            uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                            memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                    reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                    slicesToUploadMemorySize);
                        }
                                
                        asset::IImage::SBufferCopy bufferCopy;
                        bufferCopy.bufferOffset = currentUploadBufferOffset;
                        bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                        bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                        bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                        bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                        bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                        bufferCopy.imageOffset.x = region.imageOffset.x + 0u; assert(currentBlockInRow == 0);
                        bufferCopy.imageOffset.y = region.imageOffset.y + 0u; assert(currentRowInSlice == 0);
                        bufferCopy.imageOffset.z = region.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
                        bufferCopy.imageExtent.width    = imageExtent.x;
                        bufferCopy.imageExtent.height   = imageExtent.y;
                        bufferCopy.imageExtent.depth    = core::min(uploadableSlices * texelBlockDim.z, imageExtent.z);
                        bufferCopy.imageSubresource.layerCount = 1u;
                        regionsToCopy.push_back(bufferCopy);

                        addToCurrentUploadBufferOffset(slicesToUploadMemorySize);

                        currentSliceInLayer += uploadableSlices;
                        ret = true;
                    }
                            
                    if(currentSliceInLayer < imageExtentInBlocks.z)
                    {
                        bool filledAnyRowsOrBlocksInSlice = tryFillSlice();
                        if(filledAnyRowsOrBlocksInSlice)
                            ret = true;
                    }
                            
                    updateCurrentOffsets();

                    return ret;
                };

                auto tryFillRegion = [&]() -> bool
                {
                    bool ret = false;
                    uint32_t uploadableArrayLayers = availableUploadBufferMemory / eachLayerNeededMemory;
                    uint32_t remainingLayers = region.imageSubresource.layerCount - currentLayerInRegion;
                    uploadableArrayLayers = core::min(uploadableArrayLayers, remainingLayers);

                    if(uploadableArrayLayers > 0)
                    {
                        uint32_t layersToUploadMemorySize = eachLayerNeededMemory * uploadableArrayLayers;

                        if(regionBlockStrides.x != imageExtentInBlocks.x)
                        {
                            // Can't copy all rows at once, there is more padding at the end of rows, copy row by row:
                            for(uint32_t layer = 0; layer < uploadableArrayLayers; ++layer)
                            {
                                for(uint32_t z = 0; z < imageExtentInBlocks.z; ++z)
                                {
                                    for(uint32_t y = 0; y < imageExtentInBlocks.y; ++y)
                                    {
                                        auto localImageOffset = core::vector3du32_SIMD(0u, 0u + y, 0u + z, currentLayerInRegion + layer);
                                        uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                                        uint64_t offsetInUploadBuffer = currentUploadBufferOffset + layer * eachLayerNeededMemory + z * eachSliceNeededMemory + y * eachRowNeededMemory;
                                        memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                                reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                                eachRowNeededMemory);
                                    }
                                }
                            }
                        }
                        else if (regionBlockStrides.y != imageExtentInBlocks.y)
                        {
                            assert(imageOffsetInBlocks.x == 0u);
                            // Can't copy all slices at once, there is more padding at the end of slices, copy slice by slice
                                    
                            for(uint32_t layer = 0; layer < uploadableArrayLayers; ++layer)
                            {
                                for(uint32_t z = 0; z < imageExtentInBlocks.z; ++z)
                                {
                                    auto localImageOffset = core::vector3du32_SIMD(0u, 0u, 0u + z, currentLayerInRegion + layer);
                                    uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                                    uint64_t offsetInUploadBuffer = currentUploadBufferOffset + layer * eachLayerNeededMemory + z * eachSliceNeededMemory;
                                    memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                            reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                            eachSliceNeededMemory);
                                }
                            }
                        }
                        else
                        {
                            // We can copy all arrays and slices at once, because imageExtent is fit to bufferRowLength and bufferImageHeight
                            assert(imageOffsetInBlocks.x == 0u);
                            assert(imageOffsetInBlocks.y == 0u);
                            auto localImageOffset = core::vector3du32_SIMD(0u, 0u, 0u, currentLayerInRegion);
                            uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(localImageOffset, regionBlockStridesInBytes)[0];
                            uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                            memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                    reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                    layersToUploadMemorySize);
                        }
                                
                        asset::IImage::SBufferCopy bufferCopy;
                        bufferCopy.bufferOffset = currentUploadBufferOffset;
                        bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                        bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                        bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                        bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                        bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                        bufferCopy.imageOffset.x = region.imageOffset.x + 0u; assert(currentBlockInRow == 0);
                        bufferCopy.imageOffset.y = region.imageOffset.y + 0u; assert(currentRowInSlice == 0);
                        bufferCopy.imageOffset.z = region.imageOffset.z + 0u; assert(currentSliceInLayer == 0);
                        bufferCopy.imageExtent.width    = imageExtent.x;
                        bufferCopy.imageExtent.height   = imageExtent.y;
                        bufferCopy.imageExtent.depth    = imageExtent.z;
                        bufferCopy.imageSubresource.layerCount = uploadableArrayLayers;
                        regionsToCopy.push_back(bufferCopy);

                        addToCurrentUploadBufferOffset(layersToUploadMemorySize);

                        currentLayerInRegion += uploadableArrayLayers;
                        ret = true;
                    }

                    // currentLayerInRegion is respective to region.imageSubresource.baseArrayLayer so It's not in the calculations until the cmdCopy.
                    if(currentLayerInRegion < region.imageSubresource.layerCount && canTransferMipLevelsPartially)
                    {
                        bool filledAnySlicesOrRowsOrBlocksInLayer = tryFillLayer();
                        if(filledAnySlicesOrRowsOrBlocksInLayer)
                            ret = true;
                    }
                            
                    updateCurrentOffsets();

                    return ret;
                };

                    // There is remaining layers in region that needs copying
                auto remainingLayersInRegion = region.imageSubresource.layerCount - currentLayerInRegion;
                if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && remainingLayersInRegion > 0)
                {
                    bool success = tryFillRegion();
                    if(success)
                        anyTransferRecorded = true;
                    else
                        break; // not enough mem -> break
                }
                // There is remaining slices in layer that needs copying
                else if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer > 0)
                {
                    assert(canTransferMipLevelsPartially);
                    bool success = tryFillLayer();
                    if(success)
                        anyTransferRecorded = true;
                    else
                        break; // not enough mem -> break
                }
                // There is remaining rows in slice that needs copying
                else if (currentBlockInRow == 0 && currentRowInSlice > 0)
                {
                    assert(canTransferMipLevelsPartially);
                    bool success = tryFillSlice();
                    if(success)
                        anyTransferRecorded = true;
                    else
                        break; // not enough mem -> break
                }
                // There is remaining blocks in row that needs copying
                else if (currentBlockInRow > 0)
                {
                    assert(canTransferMipLevelsPartially);
                    bool success = tryFillRow();
                    if(success)
                        anyTransferRecorded = true;
                    else
                        break; // not enough mem -> break
                }
            }

            if(!regionsToCopy.empty())
            {
                cmdbuf->copyBufferToImage(m_defaultUploadBuffer.get()->getBuffer(), dstImage, dstImageLayout, regionsToCopy.size(), regionsToCopy.data());
            }

            assert(anyTransferRecorded && "uploadBufferSize is not enough to support the smallest possible transferable units to image, may be caused if your queueFam's minImageTransferGranularity is large or equal to <0,0,0>.");
            
            // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
            if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate()) {
                IDriverMemoryAllocation::MappedMemoryRange flushRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(), localOffset, uploadBufferSize);
                m_device->flushMappedMemoryRanges(1u, &flushRange);
            }
        }

        // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
        m_defaultUploadBuffer.get()->multi_free(1u, &localOffset, &uploadBufferSize, core::smart_refctd_ptr<IGPUFence>(fence), &cmdbuf); // can queue with a reset but not yet pending fence, just fine
    }
}

void IUtilities::updateImageViaStagingBuffer(
    IGPUFence* fence, IGPUQueue* queue,
    asset::ICPUBuffer const* srcBuffer, const core::SRange<const asset::IImage::SBufferCopy>& regions, video::IGPUImage* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout,
    uint32_t waitSemaphoreCount, IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore,
    const uint32_t signalSemaphoreCount, IGPUSemaphore* const* semaphoresToSignal
)
{
    core::smart_refctd_ptr<IGPUCommandPool> pool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
    core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
    m_device->createCommandBuffers(pool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
    assert(cmdbuf);
    cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
    updateImageViaStagingBuffer(cmdbuf.get(),fence,queue,srcBuffer,regions,dstImage,dstImageLayout,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore);
    cmdbuf->end();
    IGPUQueue::SSubmitInfo submit;
    submit.commandBufferCount = 1u;
    submit.commandBuffers = &cmdbuf.get();
    submit.signalSemaphoreCount = signalSemaphoreCount;
    submit.pSignalSemaphores = semaphoresToSignal;
    assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
    submit.waitSemaphoreCount = waitSemaphoreCount;
    submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
    submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
    queue->submit(1u,&submit,fence);
}

void IUtilities::updateImageViaStagingBuffer(
    IGPUQueue* queue,
    asset::ICPUBuffer const* srcBuffer, const core::SRange<const asset::IImage::SBufferCopy>& regions, video::IGPUImage* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout,
    uint32_t waitSemaphoreCount, IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore,
    const uint32_t signalSemaphoreCount, IGPUSemaphore* const* semaphoresToSignal
)
{
    auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
    updateImageViaStagingBuffer(fence.get(),queue,srcBuffer,regions,dstImage,dstImageLayout,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,signalSemaphoreCount,semaphoresToSignal);
    auto* fenceptr = fence.get();
    m_device->blockForFences(1u,&fenceptr);
}

} // namespace nbl::video