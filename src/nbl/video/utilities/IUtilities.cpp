#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/utilities/ImageRegionIterator.h"
//#include <numeric>

namespace nbl::video
{

bool IUtilities::updateImageViaStagingBuffer(
    SIntendedSubmitInfo& intendedNextSubmit, const void* srcData, asset::E_FORMAT srcFormat,
    IGPUImage* dstImage, IGPUImage::LAYOUT currentDstImageLayout,
    const std::span<const asset::IImage::SBufferCopy> regions
)
{
    if (!m_defaultUploadBuffer)
    {
        m_logger.log("no staging buffer available for upload. check `upstreamSize` passed to `IUtilities::create`",system::ILogger::ELL_ERROR);
        return false;
    }
    auto* scratch = commonTransferValidation(intendedNextSubmit);
    if (!scratch)
        return false;
   
    const auto& limits = m_device->getPhysicalDevice()->getLimits();
 
    if (regions.size() == 0)
        return false; // won't log an error cause its not one
    
    if (!srcData || !dstImage)
    {
        m_logger.log("Invalid `srcData` or `dstImage` cannot `updateImageViaStagingBuffer`.", nbl::system::ILogger::ELL_ERROR);
        return false;
    }
    

    if (dstImage->getCreationParameters().samples != asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
    {
        _NBL_TODO(); // "Erfan hasn't figured out yet how to copy to multisampled images"
        return false;
    }

    const auto& queueFamProps = m_device->getPhysicalDevice()->getQueueFamilyProperties()[intendedNextSubmit.queue->getFamilyIndex()];
    auto texelBlockInfo = asset::TexelBlockInfo(dstImage->getCreationParameters().format);
    auto minGranularity = queueFamProps.minImageTransferGranularity;
    
    assert(dstImage->getCreationParameters().format != asset::EF_UNKNOWN);
    if (srcFormat == asset::EF_UNKNOWN)
    {
        // If valid srcFormat is not provided, assume srcBuffer is laid out in memory based on dstImage format
        srcFormat = dstImage->getCreationParameters().format;
    }

    // Validate Copies from srcBuffer to dstImage with these regions
    // if the initial regions are valid then ImageRegionIterator will do it's job correctly breaking it down ;)
    // note to future self: couldn't use dstImage->validateCopies because it doesn't consider that cpubuffer will be promoted and hence it will get a validation error about size of the buffer being smaller than max accessible offset.
    bool regionsValid = true;
    for (const auto region : regions)
    {
        auto subresourceSize = dstImage->getMipSize(region.imageSubresource.mipLevel);
        if (!dstImage->validateCopyOffsetAndExtent(region.imageExtent, region.imageOffset, subresourceSize, minGranularity))
            regionsValid = false;
    }
    if (!regionsValid)
    {
        m_logger.log("Invalid regions to copy cannot `updateImageViaStagingBuffer`.", nbl::system::ILogger::ELL_ERROR);
        return false;
    }

    ImageRegionIterator regionIterator(regions, queueFamProps, srcData, srcFormat, dstImage, limits.optimalBufferCopyRowPitchAlignment);

    // TODO: Why did we settle on `/4` ? It definitely wasn't about the uint32_t size!
    // Assuming each thread can handle minImageTranferGranularitySize of texelBlocks:
    const uint32_t maxResidentImageTransferSize = core::min<uint32_t>(limits.maxResidentInvocations*minGranularity.depth*minGranularity.height*minGranularity.width*texelBlockInfo.getBlockByteSize(),m_defaultUploadBuffer->get_total_size()/4);

    core::vector<asset::IImage::SBufferCopy> regionsToCopy;

    // Worst case iterations: remaining blocks --> remaining rows --> remaining slices --> full layers
    const uint32_t maxIterations = regions.size() * 4u;

    regionsToCopy.reserve(maxIterations);

    core::vector<ILogicalDevice::MappedMemoryRange> flushRanges;
    const bool manualFlush = m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate();
    if (manualFlush)
        flushRanges.reserve(maxIterations);

    auto* uploadBuffer = m_defaultUploadBuffer.get()->getBuffer();
    // for the signal to be useful for us to let go of memory, we need to signal after transfer is finished
    const auto oldScratchStage = intendedNextSubmit.scratchSemaphore.stageMask|=asset::PIPELINE_STAGE_FLAGS::COPY_BIT;
    while (!regionIterator.isFinished())
    {
        size_t memoryNeededForRemainingRegions = regionIterator.getMemoryNeededForRemainingRegions();

        uint32_t memoryLowerBound = maxResidentImageTransferSize;
        {
            const asset::IImage::SBufferCopy & region = regions[regionIterator.getCurrentRegion()];
            const auto copyTexelStrides = regionIterator.getOptimalCopyTexelStrides(region.imageExtent);
            const auto byteStrides = texelBlockInfo.convert3DTexelStridesTo1DByteStrides(copyTexelStrides);
            memoryLowerBound = core::max(memoryLowerBound, byteStrides[1]); // max of memoryLowerBound and copy rowPitch
        }

        uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
        uint32_t maxFreeBlock = m_defaultUploadBuffer.get()->max_size();
        const uint32_t allocationSize = getAllocationSizeForStreamingBuffer(memoryNeededForRemainingRegions, m_allocationAlignmentForBufferImageCopy, maxFreeBlock, memoryLowerBound);
        // cannot use `multi_place` because of the extra padding size we could have added
        m_defaultUploadBuffer.get()->multi_allocate(std::chrono::steady_clock::now()+std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &m_allocationAlignmentForBufferImageCopy);
        bool failedAllocation = (localOffset == video::StreamingTransientDataBufferMT<>::invalid_value);

        // keep trying again
        if (failedAllocation)
        {
            if (!flushRanges.empty())
            {
                m_device->flushMappedMemoryRanges(flushRanges);
                flushRanges.clear();
            }
            const auto completed = intendedNextSubmit.getFutureScratchSemaphore();
            intendedNextSubmit.overflowSubmit(scratch);
            // first submit we respect whatever stages the user had (maybe they wanted to be notified of the completion of `nextSubmit.prevCommandBuffers`
            intendedNextSubmit.scratchSemaphore.stageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT;
            // overflowSubmit no longer blocks for the last submit to have completed, so we must do it ourselves here
            // TODO: if we cleverly overflowed BEFORE completely running out of memory (better heuristics) then we wouldn't need to do this and some CPU-GPU overlap could be achieved
            if (intendedNextSubmit.overflowCallback)
                intendedNextSubmit.overflowCallback(completed);
            m_device->blockForSemaphores({&completed,1});
            m_defaultUploadBuffer->cull_frees(); // is this even needed anymore?
            continue;
        }
        else
        {
            uint32_t currentUploadBufferOffset = localOffset;
            uint32_t availableUploadBufferMemory = allocationSize;

            regionsToCopy.clear();
            for (uint32_t d = 0u; d < maxIterations && !regionIterator.isFinished(); ++d)
            {
                asset::IImage::SBufferCopy nextRegionToCopy = {};
                if (availableUploadBufferMemory > 0u && regionIterator.advanceAndCopyToStagingBuffer(nextRegionToCopy, availableUploadBufferMemory, currentUploadBufferOffset, m_defaultUploadBuffer->getBufferPointer()))
                {
                    regionsToCopy.push_back(nextRegionToCopy);
                }
                else
                    break;
            }

            if (!regionsToCopy.empty())
                scratch->cmdbuf->copyBufferToImage(uploadBuffer, dstImage, currentDstImageLayout, regionsToCopy.size(), regionsToCopy.data());

            assert(!regionsToCopy.empty() && "allocationSize is not enough to support the smallest possible transferable units to image, may be caused if your queueFam's minImageTransferGranularity is large or equal to <0,0,0>.");
            
            // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
            if (manualFlush)
            {
                const auto consumedMemory = allocationSize - availableUploadBufferMemory;
                flushRanges.emplace_back(uploadBuffer->getBoundMemory().memory, localOffset, consumedMemory, ILogicalDevice::MappedMemoryRange::align_non_coherent_tag);
            }
        }

        // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
        m_defaultUploadBuffer.get()->multi_deallocate(1u,&localOffset,&allocationSize,intendedNextSubmit.getFutureScratchSemaphore()); // can queue with a reset but not yet pending fence, just fine
    }
    intendedNextSubmit.scratchSemaphore.stageMask = oldScratchStage;
    if (!flushRanges.empty())
        m_device->flushMappedMemoryRanges(flushRanges);
    return true;
}

bool IUtilities::downloadImageViaStagingBuffer(
    SIntendedSubmitInfo& intendedNextSubmit, const IGPUImage* srcImage, const IGPUImage::LAYOUT currentSrcImageLayout,
    void* dest, const std::span<const asset::IImage::SBufferCopy> regions
)
{
    if (!m_defaultDownloadBuffer)
    {
        m_logger.log("no staging buffer available for download. check `downstreamSize` passed to `IUtilities::create`",system::ILogger::ELL_ERROR);
        return false;
    }
    if (regions.empty())
        return false; // won't log an error cause its not one

    auto* scratch = commonTransferValidation(intendedNextSubmit);
    if (!scratch)
        return false;

    if (!srcImage || !dest)
    {
        m_logger.log("Invalid `srcImage` or `dest` cannot `downloadImageViaStagingBuffer`.", nbl::system::ILogger::ELL_ERROR);
        return false;
    }

    // VUID-vkCmdCopyImageToBuffer-srcImage-00186
    if (!srcImage->getCreationParameters().usage.hasFlags(asset::IImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT))
    {
        m_logger.log("`srcImage` has no `EUF_TRANSFER_SRC_BIT` usage flag, cannot `downloadImageViaStagingBuffer`.", nbl::system::ILogger::ELL_ERROR);
        return false;
    }

    if (srcImage->getCreationParameters().samples != asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
    {
        _NBL_TODO();
        return false;
    }

    const auto& limits = m_device->getPhysicalDevice()->getLimits();
    const auto& queueFamProps = m_device->getPhysicalDevice()->getQueueFamilyProperties()[intendedNextSubmit.queue->getFamilyIndex()];
    const auto srcImageFormat = srcImage->getCreationParameters().format;
    const auto texelBlockInfo = asset::TexelBlockInfo(srcImageFormat);
    const auto minGranularity = queueFamProps.minImageTransferGranularity;

    bool regionsValid = true;
    for (const auto region : regions)
    {
        auto subresourceSize = srcImage->getMipSize(region.imageSubresource.mipLevel);
        if (!srcImage->validateCopyOffsetAndExtent(region.imageExtent, region.imageOffset, subresourceSize, minGranularity))
            regionsValid = false;
    }
    if (!regionsValid)
    {
        m_logger.log("Invalid regions to copy cannot `downloadImageViaStagingBuffer`.", nbl::system::ILogger::ELL_ERROR);
        return false;
    }

    ImageRegionIterator regionIterator(regions, queueFamProps, srcImage, limits.optimalBufferCopyRowPitchAlignment);

    uint32_t maxRowPitch = 0u;
    for (const auto region : regions)
    {
        const auto copyTexelStrides = regionIterator.getOptimalCopyTexelStrides(region.imageExtent);
        maxRowPitch = core::max(maxRowPitch,texelBlockInfo.convert3DTexelStridesTo1DByteStrides(copyTexelStrides)[1]);
    }
    if (core::alignUp(maxRowPitch,m_allocationAlignmentForBufferImageCopy)>m_defaultDownloadBuffer->get_total_size())
    {
        m_logger.log("Download staging buffer of %u bytes cannot hold a single %u byte row, cannot `downloadImageViaStagingBuffer`.",system::ILogger::ELL_ERROR,m_defaultDownloadBuffer->get_total_size(),maxRowPitch);
        return false;
    }

    const uint32_t maxResidentImageTransferSize = core::min<uint32_t>(limits.maxResidentInvocations*minGranularity.depth*minGranularity.height*minGranularity.width*texelBlockInfo.getBlockByteSize(),m_defaultDownloadBuffer->get_total_size()/4);

    // Worst case iterations: remaining blocks --> remaining rows --> remaining slices --> full layers
    const uint32_t maxIterations = regions.size() * 4u;

    core::vector<asset::IImage::SBufferCopy> regionsToCopy;
    regionsToCopy.reserve(maxIterations);
    core::vector<asset::IImage::SBufferCopy> parentRegions;
    parentRegions.reserve(maxIterations);

    // for the signal to be useful for us to execute the data consumer callback, the signal must happen after the copy is done
    const auto oldScratchStage = intendedNextSubmit.scratchSemaphore.stageMask|=asset::PIPELINE_STAGE_FLAGS::COPY_BIT;
    while (!regionIterator.isFinished())
    {
        const size_t memoryNeededForRemainingRegions = regionIterator.getMemoryNeededForRemainingRegions();

        uint32_t memoryLowerBound = maxResidentImageTransferSize;
        {
            const asset::IImage::SBufferCopy & region = regions[regionIterator.getCurrentRegion()];
            const auto copyTexelStrides = regionIterator.getOptimalCopyTexelStrides(region.imageExtent);
            const auto byteStrides = texelBlockInfo.convert3DTexelStridesTo1DByteStrides(copyTexelStrides);
            memoryLowerBound = core::max(memoryLowerBound, byteStrides[1]); // max of memoryLowerBound and copy rowPitch
        }

        uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
        const uint32_t maxFreeBlock = m_defaultDownloadBuffer->max_size();
        const uint32_t allocationSize = getAllocationSizeForStreamingBuffer(memoryNeededForRemainingRegions, m_allocationAlignmentForBufferImageCopy, maxFreeBlock, memoryLowerBound);
        m_defaultDownloadBuffer->multi_allocate(std::chrono::steady_clock::now()+std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &m_allocationAlignmentForBufferImageCopy);

        // keep trying again
        if (localOffset == video::StreamingTransientDataBufferMT<>::invalid_value)
        {
            const auto completed = intendedNextSubmit.getFutureScratchSemaphore();
            if (intendedNextSubmit.overflowSubmit(scratch)!=IQueue::RESULT::SUCCESS)
            {
                m_logger.log("Overflow submit failed, cannot `downloadImageViaStagingBuffer`.",system::ILogger::ELL_ERROR);
                intendedNextSubmit.scratchSemaphore.stageMask = oldScratchStage;
                return false;
            }
            // first submit we respect whatever stages the user had (maybe they wanted to be notified of the completion of `nextSubmit.prevCommandBuffers`
            intendedNextSubmit.scratchSemaphore.stageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT;
            // overflowSubmit no longer blocks for the last submit to have completed, so we must do it ourselves here
            if (intendedNextSubmit.overflowCallback)
                intendedNextSubmit.overflowCallback(completed);
            if (m_device->blockForSemaphores({&completed,1})!=ISemaphore::WAIT_RESULT::SUCCESS)
            {
                m_logger.log("Waiting for the overflow submit failed, cannot `downloadImageViaStagingBuffer`.",system::ILogger::ELL_ERROR);
                intendedNextSubmit.scratchSemaphore.stageMask = oldScratchStage;
                return false;
            }
            m_defaultDownloadBuffer->cull_frees();
            continue;
        }

        uint32_t currentDownloadBufferOffset = localOffset;
        uint32_t availableDownloadBufferMemory = allocationSize;

        regionsToCopy.clear();
        parentRegions.clear();
        for (uint32_t d = 0u; d < maxIterations && !regionIterator.isFinished(); ++d)
        {
            const asset::IImage::SBufferCopy parentRegion = regions[regionIterator.getCurrentRegion()];
            asset::IImage::SBufferCopy nextRegionToCopy = {};
            if (availableDownloadBufferMemory > 0u && regionIterator.advance(nextRegionToCopy, availableDownloadBufferMemory, currentDownloadBufferOffset))
            {
                regionsToCopy.push_back(nextRegionToCopy);
                parentRegions.push_back(parentRegion);
            }
            else
                break;
        }

        bool copyRecorded = false;
        bool recorded = false;
        if (regionsToCopy.empty())
            m_logger.log("Allocation of %u bytes cannot hold the smallest transferable unit, check the queue family's `minImageTransferGranularity`, cannot `downloadImageViaStagingBuffer`.",system::ILogger::ELL_ERROR,allocationSize);
        else
        {
            copyRecorded = scratch->cmdbuf->copyImageToBuffer(srcImage, currentSrcImageLayout, m_defaultDownloadBuffer->getBuffer(), regionsToCopy.size(), regionsToCopy.data());
            if (copyRecorded)
            {
                const asset::SMemoryBarrier hostRead = {
                    .srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
                    .srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
                    .dstStageMask = asset::PIPELINE_STAGE_FLAGS::HOST_BIT,
                    .dstAccessMask = asset::ACCESS_FLAGS::HOST_READ_BIT
                };
                recorded = scratch->cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={&hostRead,1}});
            }
            if (!recorded)
                m_logger.log("Failed to record the image to staging copy, cannot `downloadImageViaStagingBuffer`.",system::ILogger::ELL_ERROR);
        }
        if (!recorded)
        {
            // the copy may already be recorded, so the block can only be reused once the scratch semaphore signals
            if (copyRecorded)
                m_defaultDownloadBuffer->multi_deallocate(1u,&localOffset,&allocationSize,intendedNextSubmit.getFutureScratchSemaphore());
            else
                m_defaultDownloadBuffer->multi_deallocate(1u,&localOffset,&allocationSize);
            intendedNextSubmit.scratchSemaphore.stageMask = oldScratchStage;
            return false;
        }

        const uint32_t consumedMemory = allocationSize - availableDownloadBufferMemory;
        auto dataConsumer = core::make_smart_refctd_ptr<CDownstreamingDataConsumer>(
            IDeviceMemoryAllocation::MemoryRange(localOffset,consumedMemory),
            [dest,localOffset,srcImageFormat,subRegions=regionsToCopy,dstRegions=parentRegions](const size_t dstOffset, const void* srcPtr, const size_t size)->void
            {
                const asset::TexelBlockInfo blockInfo(srcImageFormat);
                const uint32_t blockByteSize = blockInfo.getBlockByteSize();
                for (size_t i=0; i<subRegions.size(); i++)
                {
                    const asset::IImage::SBufferCopy& sub = subRegions[i];
                    const asset::IImage::SBufferCopy& parent = dstRegions[i];

                    const auto srcByteStrides = sub.getByteStrides(blockInfo);
                    const auto dstByteStrides = parent.getByteStrides(blockInfo);
                    const auto extentInBlocks = blockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(sub.imageExtent.width,sub.imageExtent.height,sub.imageExtent.depth));
                    const auto localBlockOffset = blockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(
                        sub.imageOffset.x-parent.imageOffset.x,
                        sub.imageOffset.y-parent.imageOffset.y,
                        sub.imageOffset.z-parent.imageOffset.z
                    ));
                    const uint32_t localLayer = sub.imageSubresource.baseArrayLayer-parent.imageSubresource.baseArrayLayer;

                    const size_t rowByteSize = size_t(extentInBlocks.x)*blockByteSize;
                    const uint8_t* srcBase = reinterpret_cast<const uint8_t*>(srcPtr)+(sub.bufferOffset-localOffset);
                    uint8_t* dstBase = reinterpret_cast<uint8_t*>(dest)+parent.bufferOffset;
                    for (uint32_t l=0u; l<sub.imageSubresource.layerCount; l++)
                    for (uint32_t z=0u; z<extentInBlocks.z; z++)
                    for (uint32_t y=0u; y<extentInBlocks.y; y++)
                    {
                        const uint8_t* src = srcBase+size_t(l)*srcByteStrides[3]+size_t(z)*srcByteStrides[2]+size_t(y)*srcByteStrides[1];
                        assert(src+rowByteSize<=reinterpret_cast<const uint8_t*>(srcPtr)+size);
                        uint8_t* dst = dstBase+asset::IImage::SBufferCopy::getLocalByteOffset(
                            core::vector4du32_SIMD(localBlockOffset.x,localBlockOffset.y+y,localBlockOffset.z+z,localLayer+l),dstByteStrides
                        );
                        memcpy(dst,src,rowByteSize);
                    }
                }
            },
            core::smart_refctd_ptr<IGPUCommandBuffer>(scratch->cmdbuf),
            m_defaultDownloadBuffer.get()
        );
        // this doesn't actually free the memory, the memory is queued up to be freed only after the `scratchSemaphore` reaches a value a future submit will signal
        m_defaultDownloadBuffer->multi_deallocate(1u,&localOffset,&allocationSize,intendedNextSubmit.getFutureScratchSemaphore(),&dataConsumer.get());
    }
    intendedNextSubmit.scratchSemaphore.stageMask = oldScratchStage;
    return true;
}

} // namespace nbl::video