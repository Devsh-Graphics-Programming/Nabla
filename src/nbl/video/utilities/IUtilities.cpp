#include "nbl/video/utilities/IUtilities.h"
#include "nbl/asset/filters/CConvertFormatImageFilter.h"
#include <numeric>

namespace nbl::video
{
IGPUQueue::SSubmitInfo IUtilities::updateImageViaStagingBuffer(
    asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, asset::IImage::E_LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
    IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo intendedNextSubmit)
{
    if(!intendedNextSubmit.isValid() || intendedNextSubmit.commandBufferCount <= 0u)
    {
        // TODO: log error -> intendedNextSubmit is invalid
        assert(false);
        return intendedNextSubmit;
    }

    // Use the last command buffer in intendedNextSubmit, it should be in recording state
    auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount-1];

    assert(cmdbuf->getState() == IGPUCommandBuffer::ES_RECORDING && cmdbuf->isResettable());
    assert(cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT));
   
    const auto& limits = m_device->getPhysicalDevice()->getLimits();
 
    if (regions.size() == 0)
        return intendedNextSubmit;
    
    if (cmdbuf == nullptr || submissionFence == nullptr || submissionQueue == nullptr || dstImage == nullptr || (srcBuffer == nullptr || srcBuffer->getPointer() == nullptr))
    {
        assert(false);
        return intendedNextSubmit;
    }
    
    auto* cmdpool = cmdbuf->getPool();
    assert(cmdpool->getQueueFamilyIndex()==submissionQueue->getFamilyIndex());
    if (dstImage->getCreationParameters().samples != asset::IImage::ESCF_1_BIT)
    {
        _NBL_TODO(); // "Erfan hasn't figured out yet how to copy to multisampled images"
        return intendedNextSubmit;
    }

    auto texelBlockInfo = asset::TexelBlockInfo(dstImage->getCreationParameters().format);
    auto queueFamProps = m_device->getPhysicalDevice()->getQueueFamilyProperties()[submissionQueue->getFamilyIndex()];
    auto minImageTransferGranularity = queueFamProps.minImageTransferGranularity;
    
    assert(dstImage->getCreationParameters().format != asset::EF_UNKNOWN);
    if (srcFormat == asset::EF_UNKNOWN)
    {
        // If valid srcFormat is not provided, assume srcBuffer is laid out in memory based on dstImage format
        srcFormat = dstImage->getCreationParameters().format;
    }

    ImageRegionIterator regionIterator = ImageRegionIterator(regions, queueFamProps, srcBuffer, srcFormat, dstImage);

    // Assuming each thread can handle minImageTranferGranularitySize of texelBlocks:
    const uint32_t maxResidentImageTransferSize = limits.maxResidentInvocations * texelBlockInfo.getBlockByteSize() * (minImageTransferGranularity.width * minImageTransferGranularity.height * minImageTransferGranularity.depth); 

    core::vector<asset::IImage::SBufferCopy> regionsToCopy;

    // Worst case iterations: remaining blocks --> remaining rows --> remaining slices --> full layers
    const uint32_t maxIterations = regions.size() * 4u;

    regionsToCopy.reserve(maxIterations);

    while (!regionIterator.isFinished())
    {
        size_t memoryNeededForRemainingRegions = regionIterator.getMemoryNeededForRemainingRegions();

        uint32_t memoryLowerBound = maxResidentImageTransferSize;
        {
            const asset::IImage::SBufferCopy & region = regions[regionIterator.getCurrentRegion()];
            auto imageExtent = core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth);
            auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(imageExtent);
            auto imageExtentBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);
            memoryLowerBound = core::max(memoryLowerBound, imageExtentBlockStridesInBytes[1]); // rowPitch = imageExtentBlockStridesInBytes[1]
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
            // but first submit the already buffered up copies and whatever previously recorded into the command buffer
            cmdbuf->end();
            IGPUQueue::SSubmitInfo submit = intendedNextSubmit;
            submit.signalSemaphoreCount = 0u;
            submit.pSignalSemaphores = nullptr;
            assert(submit.isValid());
            submissionQueue->submit(1u, &submit, submissionFence);
            m_device->blockForFences(1u, &submissionFence);
            intendedNextSubmit.commandBufferCount = 1u;
            intendedNextSubmit.commandBuffers = &cmdbuf;
            intendedNextSubmit.waitSemaphoreCount = 0u;
            intendedNextSubmit.pWaitSemaphores = nullptr;
            intendedNextSubmit.pWaitDstStageMask = nullptr;
            // before resetting we need poll all events in the allocator's deferred free list
            m_defaultUploadBuffer->cull_frees();
            // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
            m_device->resetFences(1u, &submissionFence);
            cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
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
            {
                cmdbuf->copyBufferToImage(m_defaultUploadBuffer.get()->getBuffer(), dstImage, currentDstImageLayout, regionsToCopy.size(), regionsToCopy.data());
            }

            assert(!regionsToCopy.empty() && "allocationSize is not enough to support the smallest possible transferable units to image, may be caused if your queueFam's minImageTransferGranularity is large or equal to <0,0,0>.");
            
            // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
            if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate())
            {
                const auto consumedMemory = allocationSize - availableUploadBufferMemory;
                auto flushRange = AlignedMappedMemoryRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(), localOffset, consumedMemory, limits.nonCoherentAtomSize);
                m_device->flushMappedMemoryRanges(1u, &flushRange);
            }
        }

        // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
        m_defaultUploadBuffer.get()->multi_deallocate(1u, &localOffset, &allocationSize, core::smart_refctd_ptr<IGPUFence>(submissionFence), &cmdbuf); // can queue with a reset but not yet pending fence, just fine
    }
    return intendedNextSubmit;
}

void IUtilities::updateImageViaStagingBufferAutoSubmit(
    asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, asset::IImage::E_LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
    IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo submitInfo
)
{
    if(!submitInfo.isValid())
    {
        // TODO: log error
        assert(false);
        return;
    }

    CSubmitInfoPatcher submitInfoPatcher;
    submitInfoPatcher.patchAndBegin(submitInfo, m_device, submissionQueue->getFamilyIndex());
    submitInfo = updateImageViaStagingBuffer(srcBuffer,srcFormat,dstImage,currentDstImageLayout,regions,submissionQueue,submissionFence,submitInfo);
    submitInfoPatcher.end();

    assert(submitInfo.isValid());
    submissionQueue->submit(1u,&submitInfo,submissionFence);
}

void IUtilities::updateImageViaStagingBufferAutoSubmit(
    asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, asset::IImage::E_LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
    IGPUQueue* submissionQueue, const IGPUQueue::SSubmitInfo& submitInfo
)
{
    if(!submitInfo.isValid())
    {
        // TODO: log error
        assert(false);
        return;
    }

    auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
    updateImageViaStagingBufferAutoSubmit(srcBuffer,srcFormat,dstImage,currentDstImageLayout,regions,submissionQueue,fence.get(),submitInfo);
    m_device->blockForFences(1u,&fence.get());
}

ImageRegionIterator::ImageRegionIterator(
    const core::SRange<const asset::IImage::SBufferCopy>& copyRegions,
    IPhysicalDevice::SQueueFamilyProperties queueFamilyProps,
    asset::ICPUBuffer const* srcBuffer,
    asset::E_FORMAT srcImageFormat,
    video::IGPUImage* const dstImage
)
    : regions(copyRegions)
    , minImageTransferGranularity(queueFamilyProps.minImageTransferGranularity)
    , srcBuffer(srcBuffer)
    , dstImage(dstImage)
    , srcImageFormat(srcImageFormat)
    , currentBlockInRow(0u)
    , currentRowInSlice(0u)
    , currentSliceInLayer(0u)
    , currentLayerInRegion(0u)
    , currentRegion(0u)
{
    dstImageFormat = dstImage->getCreationParameters().format;
    if(srcImageFormat == asset::EF_UNKNOWN)
        srcImageFormat = dstImageFormat;
    asset::TexelBlockInfo dstImageTexelBlockInfo(dstImageFormat);

    // bufferOffsetAlignment:
        // [x] If Depth/Stencil -> must be multiple of 4
        // [ ] If multi-planar -> bufferOffset must be a multiple of the element size of the compatible format for the aspectMask of imagesubresource
        // [x] If Queue doesn't support GRAPHICS_BIT or COMPUTE_BIT ->  must be multiple of 4
        // [x] bufferOffset must be a multiple of texel block size in bytes
    bufferOffsetAlignment = dstImageTexelBlockInfo.getBlockByteSize(); // can be non power of two
    if(asset::isDepthOrStencilFormat(dstImageFormat))
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);

    bool queueSupportsCompute = queueFamilyProps.queueFlags.hasFlags(IPhysicalDevice::EQF_COMPUTE_BIT);
    bool queueSupportsGraphics = queueFamilyProps.queueFlags.hasFlags(IPhysicalDevice::EQF_GRAPHICS_BIT);
    if((queueSupportsGraphics || queueSupportsCompute) == false)
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);
    // TODO: Need to have a function to get equivalent format of the specific plane of this format (in aspectMask)
    // if(asset::isPlanarFormat(dstImageFormat->getCreationParameters().format))
        
    // Queues supporting graphics and/or compute operations must report (1,1,1) in minImageTransferGranularity, meaning that there are no additional restrictions on the granularity of image transfer operations for these queues.
    // Other queues supporting image transfer operations are only required to support whole mip level transfers, thus minImageTransferGranularity for queues belonging to such queue families may be (0,0,0)
    canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);
}

size_t ImageRegionIterator::getMemoryNeededForRemainingRegions() const
{
    asset::TexelBlockInfo dstImageTexelBlockInfo(dstImageFormat);
    assert(dstImageTexelBlockInfo.getBlockByteSize()>0u);
    auto texelBlockDim = dstImageTexelBlockInfo.getDimension();
    uint32_t memoryNeededForRemainingRegions = 0ull;
    
    // We want to first roundUp to bufferOffsetAlignment everytime we increment, because the incrementation here correspond a single copy command that needs it's bufferOffset to be aligned correctly (assuming enough memory).
    auto incrementMemoryNeeded = [&](const uint32_t size)
    {
        memoryNeededForRemainingRegions = core::roundUp(memoryNeededForRemainingRegions, bufferOffsetAlignment);
        memoryNeededForRemainingRegions += size;
    };

    for (uint32_t i = currentRegion; i < regions.size(); ++i)
    {
        const asset::IImage::SBufferCopy & region = regions[i];

        auto subresourceSize = dstImage->getMipSize(region.imageSubresource.mipLevel);

        // Validate Region, TODO: move these to IGPUImage::validateCopies and call them on every region at the beginning

        assert(static_cast<uint32_t>(region.imageSubresource.aspectMask) != 0u);
        assert(core::isPoT(static_cast<uint32_t>(region.imageSubresource.aspectMask)) && "region.aspectMask should only have a single bit set.");
        
        // canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);
        if (canTransferMipLevelsPartially)
        {
            // region.imageOffset.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size
            bool isImageOffsetAlignmentValid =
                (region.imageOffset.x % (minImageTransferGranularity.width * texelBlockDim.x) == 0) &&
                (region.imageOffset.y % (minImageTransferGranularity.height * texelBlockDim.y) == 0) &&
                (region.imageOffset.z % (minImageTransferGranularity.depth * texelBlockDim.z) == 0);
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
        }
        else
        {
            assert(region.imageOffset.x == 0 && region.imageOffset.y == 0 && region.imageOffset.z == 0);
            assert(region.imageExtent.width == subresourceSize.x && region.imageExtent.height == subresourceSize.y && region.imageExtent.depth == subresourceSize.z);
        }


        auto imageExtent = core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth);
        auto imageExtentInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(imageExtent);
        auto imageExtentBlockStridesInBytes = dstImageTexelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);

        if(i == currentRegion)
        {
            auto remainingBlocksInRow = imageExtentInBlocks.x - currentBlockInRow;
            auto remainingRowsInSlice = imageExtentInBlocks.y - currentRowInSlice;
            auto remainingSlicesInLayer = imageExtentInBlocks.z - currentSliceInLayer;
            auto remainingLayersInRegion = region.imageSubresource.layerCount - currentLayerInRegion;

            if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && remainingLayersInRegion > 0)
            {
                incrementMemoryNeeded(imageExtentBlockStridesInBytes[3] * remainingLayersInRegion);
            }
            else if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer > 0)
            {
                incrementMemoryNeeded(imageExtentBlockStridesInBytes[2] * remainingSlicesInLayer);
                if (remainingLayersInRegion > 1u)
                    incrementMemoryNeeded(imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u));
            }
            else if (currentBlockInRow == 0 && currentRowInSlice > 0)
            {
                incrementMemoryNeeded(imageExtentBlockStridesInBytes[1] * remainingRowsInSlice);

                if(remainingSlicesInLayer > 1u)
                    incrementMemoryNeeded(imageExtentBlockStridesInBytes[2] * (remainingSlicesInLayer - 1u));
                if(remainingLayersInRegion > 1u)
                    incrementMemoryNeeded(imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u));
            }
            else if (currentBlockInRow > 0)
            {
                // want to first fill the remaining blocks in current row
                incrementMemoryNeeded(imageExtentBlockStridesInBytes[0] * remainingBlocksInRow);
                // then fill the remaining rows in current slice
                if(remainingRowsInSlice > 1u)
                    incrementMemoryNeeded(imageExtentBlockStridesInBytes[1] * (remainingRowsInSlice - 1u));
                // then fill the remaining slices in current layer
                if(remainingSlicesInLayer > 1u)
                    incrementMemoryNeeded(imageExtentBlockStridesInBytes[2] * (remainingSlicesInLayer - 1u));
                // then fill the remaining layers in current region
                if(remainingLayersInRegion > 1u)
                    incrementMemoryNeeded(imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u));
            }
        }
        else
        {
            // we want to fill the whole layers in the region
            incrementMemoryNeeded(imageExtentBlockStridesInBytes[3] * region.imageSubresource.layerCount); // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z * region.imageSubresource.layerCount
        }
    }
    return memoryNeededForRemainingRegions;
}

// These Swizzles makes sure copying from srcFormat image to promotedFormat image is consistent and extra "unused" channels will be ZERO and alpha will be ONE
struct FourComponentSwizzle
{
    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

        reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];
        reinterpret_cast<out_t*>(out)[1u] = reinterpret_cast<const in_t*>(in)[1u];
        reinterpret_cast<out_t*>(out)[2u] = reinterpret_cast<const in_t*>(in)[2u];
        reinterpret_cast<out_t*>(out)[3u] = reinterpret_cast<const in_t*>(in)[3u];
    }
};
struct ThreeComponentSwizzle
{
    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

        reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];
        reinterpret_cast<out_t*>(out)[1u] = reinterpret_cast<const in_t*>(in)[1u];
        reinterpret_cast<out_t*>(out)[2u] = reinterpret_cast<const in_t*>(in)[2u];
        reinterpret_cast<out_t*>(out)[3u] = static_cast<in_t>(1);
    }
};
struct TwoComponentSwizzle
{
    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

        reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];
        reinterpret_cast<out_t*>(out)[1u] = reinterpret_cast<const in_t*>(in)[1u];
        reinterpret_cast<out_t*>(out)[2u] = static_cast<in_t>(0);
        reinterpret_cast<out_t*>(out)[3u] = static_cast<in_t>(1);
    }
};
struct OneComponentSwizzle
{
    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

        reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];
        reinterpret_cast<out_t*>(out)[1u] = static_cast<in_t>(0);
        reinterpret_cast<out_t*>(out)[2u] = static_cast<in_t>(0);
        reinterpret_cast<out_t*>(out)[3u] = static_cast<in_t>(1);
    }
};

template<typename Filter>
bool performCopyUsingImageFilter(
    const core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage,
    const core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage,
    const asset::IImage::SBufferCopy& region)
{
    Filter filter;
    typename Filter::state_type state = {};
    state.extent = region.imageExtent;
    state.layerCount = region.imageSubresource.layerCount;
    state.inImage = inCPUImage.get();
    state.outImage = outCPUImage.get();
    state.inOffsetBaseLayer = core::vectorSIMDu32(0u);
    state.outOffsetBaseLayer = core::vectorSIMDu32(0u);
    state.inMipLevel = 0u;
    state.outMipLevel = 0u;

    if (filter.execute(core::execution::par_unseq, &state))
        return true;
    else
        return false;
}

bool performCopy(
    asset::E_FORMAT srcImageFormat,
    asset::E_FORMAT dstImageFormat,
    const core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage,
    const core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage,
    const asset::IImage::SBufferCopy& region)
{
    // In = srcBuffer, Out = stagingBuffer
    if (srcImageFormat == dstImageFormat)
    {
        return performCopyUsingImageFilter<asset::CCopyImageFilter>(inCPUImage, outCPUImage, region);
    }
    else
    {
        auto srcChannelCount = asset::getFormatChannelCount(srcImageFormat);
        if (srcChannelCount == 1u)
            performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, OneComponentSwizzle>>(inCPUImage, outCPUImage, region);
        else if (srcChannelCount == 2u)
            performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, TwoComponentSwizzle>>(inCPUImage, outCPUImage, region);
        else if (srcChannelCount == 3u)
            performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, ThreeComponentSwizzle>>(inCPUImage, outCPUImage, region);
        else
            performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, FourComponentSwizzle>>(inCPUImage, outCPUImage, region);
    }
}

bool ImageRegionIterator::advanceAndCopyToStagingBuffer(asset::IImage::SBufferCopy& regionToCopyNext, uint32_t& availableMemory, uint32_t& stagingBufferOffset, void* stagingBufferPointer)
{
    if(isFinished())
        return false;
        
    auto addToCurrentUploadBufferOffset = [&](uint32_t size) -> bool 
    {
        const auto initialOffset = stagingBufferOffset;
        stagingBufferOffset = core::roundUp(stagingBufferOffset, bufferOffsetAlignment);
        stagingBufferOffset += size;
        const auto consumedMemory = stagingBufferOffset - initialOffset;
        if(consumedMemory <= availableMemory)
        {
            availableMemory -= consumedMemory;
            return true;
        }
        else
        {
            return false;
        }
    };

    // early out: checking initial alignment of stagingBufferOffset and if any memory will be left after alignment
    if(!addToCurrentUploadBufferOffset(0u))
    {
        return false;
    }

    // ! Current Region that may break down into smaller regions (the first smaller region is nextRegionToCopy)
    const asset::IImage::SBufferCopy & mainRegion = regions[currentRegion];
        
    asset::TexelBlockInfo srcImageTexelBlockInfo(srcImageFormat);
    asset::TexelBlockInfo dstImageTexelBlockInfo(dstImageFormat);
        
    const core::vector4du32_SIMD srcBufferByteStrides = mainRegion.getByteStrides(srcImageTexelBlockInfo);

    // ! We only need subresourceSize for validations and assertions about minImageTransferGranularity because granularity requirements can be ignored if region fits against the right corner of the subresource (described in more detail below)
    const auto subresourceSize = dstImage->getMipSize(mainRegion.imageSubresource.mipLevel);
    const auto subresourceSizeInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(subresourceSize);
        
    // regionBlockStrides = <BufferRowLengthInBlocks, BufferImageHeightInBlocks, ImageDepthInBlocks>
    const auto regionBlockStrides = mainRegion.getBlockStrides(dstImageTexelBlockInfo);
    // regionBlockStridesInBytes = <BlockByteSize,
    //                              BlockBytesSize * BufferRowLengthInBlocks,
    //                              BlockBytesSize * BufferRowLengthInBlocks * BufferImageHeightInBlocks,
    //                              BlockBytesSize * BufferRowLengthInBlocks * BufferImageHeightInBlocks * ImageDepthInBlocks>
    const core::vector4du32_SIMD regionBlockStridesInBytes = dstImageTexelBlockInfo.convert3DBlockStridesTo1DByteStrides(regionBlockStrides);
    auto texelBlockDim = dstImageTexelBlockInfo.getDimension();

    // ! Don't confuse imageExtent with subresourceSize, imageExtent is the extent of the main region to copy and the subresourceSize is the actual size of dstImage 
    const auto imageExtent = core::vector3du32_SIMD(mainRegion.imageExtent.width, mainRegion.imageExtent.height, mainRegion.imageExtent.depth);
    const auto imageOffset = core::vector3du32_SIMD(mainRegion.imageOffset.x, mainRegion.imageOffset.y, mainRegion.imageOffset.z);
    const auto imageOffsetInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(imageOffset);
    const auto imageExtentInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(imageExtent);
    const core::vector4du32_SIMD imageExtentBlockStridesInBytes = dstImageTexelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);
             
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
        if(currentLayerInRegion >= mainRegion.imageSubresource.layerCount) 
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

    // There is remaining layers in region that needs copying
    uint32_t uploadableArrayLayers = availableMemory / eachLayerNeededMemory;
    uint32_t remainingLayers = mainRegion.imageSubresource.layerCount - currentLayerInRegion;
    uploadableArrayLayers = core::min(uploadableArrayLayers, remainingLayers);
    // A: There is remaining layers left in region -> Copy Slices (Depths)
    uint32_t uploadableSlices = availableMemory / eachSliceNeededMemory;
    uint32_t remainingSlices = imageExtentInBlocks.z - currentSliceInLayer;
    uploadableSlices = core::min(uploadableSlices, remainingSlices);
    if(uploadableSlices > 0 && minImageTransferGranularity.depth > 1u && (imageOffsetInBlocks.z + currentSliceInLayer + uploadableSlices) < subresourceSizeInBlocks.z)
        uploadableSlices = core::alignDown(uploadableSlices, minImageTransferGranularity.depth);
    // B: There is remaining slices left in layer -> Copy Rows
    uint32_t uploadableRows = availableMemory / eachRowNeededMemory;
    uint32_t remainingRows = imageExtentInBlocks.y - currentRowInSlice;
    uploadableRows = core::min(uploadableRows, remainingRows);
    if(uploadableRows > 0 && minImageTransferGranularity.height > 1u && (imageOffsetInBlocks.y + currentRowInSlice + uploadableRows) < subresourceSizeInBlocks.y)
        uploadableRows = core::alignDown(uploadableRows, minImageTransferGranularity.height);
    // C: There is remaining slices left in layer -> Copy Blocks
    uint32_t uploadableBlocks = availableMemory / eachBlockNeededMemory;
    uint32_t remainingBlocks = imageExtentInBlocks.x - currentBlockInRow;
    uploadableBlocks = core::min(uploadableBlocks, remainingBlocks);
    if(uploadableBlocks > 0 && minImageTransferGranularity.width > 1u && (imageOffsetInBlocks.x + currentBlockInRow + uploadableBlocks) < subresourceSizeInBlocks.x)
        uploadableBlocks = core::alignDown(uploadableBlocks, minImageTransferGranularity.width);

    // ! Function to create mock cpu images that can go into image filters for copying/converting
    auto createMockInOutCPUImagesForFilter = [&](core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage, core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage, const size_t outCPUBufferSize) -> void
    {        
        /*
            We have to first construct two `ICPUImage`s from each of those buffers `inCPUImage` and `outCPUImage`
            Then we will create fake ICPUBuffers that point to srcBuffer and stagingBuffer with correct offsets
            Then we have to set the buffer and regions for each one of those ICPUImages using setBufferAndRegions
            Finally we fill the filter state and `execute` which require in/out CPUImages
        */
            
        auto dstImageParams = dstImage->getCreationParameters();

        // inCPUImage is an image matching the params of dstImage but with the extents and layer count of the current region being copied and mipLevel 1u and the format being srcImageFormat
        // the buffer of this image is set to (srcBuffer+Offset) and the related region is set to cover the whole copy region (offset from 0)
        {
            auto inCpuImageRegionsDynArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1);
            auto& inCpuImageRegion = inCpuImageRegionsDynArray->front();
            inCpuImageRegion = {};
            inCpuImageRegion.bufferOffset = 0u;
            inCpuImageRegion.bufferRowLength = mainRegion.bufferRowLength;
            inCpuImageRegion.bufferImageHeight = mainRegion.bufferImageHeight;
            inCpuImageRegion.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
            inCpuImageRegion.imageSubresource.mipLevel = 0u;
            inCpuImageRegion.imageSubresource.baseArrayLayer = 0u;
            inCpuImageRegion.imageOffset.x = 0u;
            inCpuImageRegion.imageOffset.y = 0u;
            inCpuImageRegion.imageOffset.z = 0u;
            inCpuImageRegion.imageExtent.width    = regionToCopyNext.imageExtent.width;
            inCpuImageRegion.imageExtent.height   = regionToCopyNext.imageExtent.height;
            inCpuImageRegion.imageExtent.depth    = regionToCopyNext.imageExtent.depth;
            inCpuImageRegion.imageSubresource.layerCount = core::max(regionToCopyNext.imageSubresource.layerCount, 1u);

            auto localImageOffset = core::vector4du32_SIMD(currentBlockInRow, currentRowInSlice, currentSliceInLayer, currentLayerInRegion);
            uint64_t offsetInCPUBuffer = mainRegion.bufferOffset + core::dot(localImageOffset, srcBufferByteStrides)[0];
            uint8_t* inCpuBufferPointer = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(srcBuffer->getPointer()) + offsetInCPUBuffer);
            asset::ICPUImage::SCreationParams inCPUImageParams = dstImageParams;
            inCPUImageParams.flags = asset::IImage::ECF_NONE; // Because we may want to write to first few layers of CUBEMAP (<6) but it's not valid to create an Cube ICPUImage with less that 6 layers.
            inCPUImageParams.format = srcImageFormat;
            inCPUImageParams.extent = regionToCopyNext.imageExtent;
            inCPUImageParams.arrayLayers = regionToCopyNext.imageSubresource.layerCount;
            inCPUImageParams.mipLevels = 1u;
            inCPUImage = asset::ICPUImage::create(std::move(inCPUImageParams));
            assert(inCPUImage);
            core::smart_refctd_ptr<asset::ICPUBuffer> inCPUBuffer = core::make_smart_refctd_ptr< asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(srcBuffer->getSize(), inCpuBufferPointer, core::adopt_memory);
            inCPUImage->setBufferAndRegions(std::move(inCPUBuffer), inCpuImageRegionsDynArray);
            assert(inCPUImage->getBuffer());
            assert(inCPUImage->getRegions().size() > 0u);
        }

        // outCPUImage is an image matching the params of dstImage but with the extents and layer count of the current region being copied and mipLevel 1u
        // the buffer of this image is set to (stagingBufferPointer + stagingBufferOffset) and the related region is set to cover the whole copy region (offset from 0)
        {
            auto outCpuImageRegionsDynArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1);
            auto& outCpuImageRegion = outCpuImageRegionsDynArray->front();
            outCpuImageRegion = {};
            outCpuImageRegion.bufferOffset = 0u;
            outCpuImageRegion.bufferRowLength = regionToCopyNext.bufferRowLength;
            outCpuImageRegion.bufferImageHeight = regionToCopyNext.bufferImageHeight;
            outCpuImageRegion.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
            outCpuImageRegion.imageSubresource.mipLevel = 0u;
            outCpuImageRegion.imageSubresource.baseArrayLayer = 0u;
            outCpuImageRegion.imageOffset.x = 0u;
            outCpuImageRegion.imageOffset.y = 0u;
            outCpuImageRegion.imageOffset.z = 0u;
            outCpuImageRegion.imageExtent.width    = regionToCopyNext.imageExtent.width;
            outCpuImageRegion.imageExtent.height   = regionToCopyNext.imageExtent.height;
            outCpuImageRegion.imageExtent.depth    = regionToCopyNext.imageExtent.depth;
            outCpuImageRegion.imageSubresource.layerCount = core::max(regionToCopyNext.imageSubresource.layerCount, 1u);

            asset::ICPUImage::SCreationParams outCPUImageParams = dstImageParams;
            uint8_t* outCpuBufferPointer = reinterpret_cast<uint8_t*>(stagingBufferPointer) + stagingBufferOffset;
            outCPUImageParams.flags = asset::IImage::ECF_NONE; // Because we may want to write to first few layers of CUBEMAP (<6) but it's not valid to create an Cube ICPUImage with less that 6 layers.
            outCPUImageParams.extent = regionToCopyNext.imageExtent;
            outCPUImageParams.arrayLayers = regionToCopyNext.imageSubresource.layerCount;
            outCPUImageParams.mipLevels = 1u;
            outCPUImage = asset::ICPUImage::create(std::move(outCPUImageParams));
            assert(outCPUImage);
            core::smart_refctd_ptr<asset::ICPUBuffer> outCPUBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(outCPUBufferSize, outCpuBufferPointer, core::adopt_memory);
            outCPUImage->setBufferAndRegions(std::move(outCPUBuffer), outCpuImageRegionsDynArray);
            assert(outCPUImage->getBuffer());
            assert(outCPUImage->getRegions().size() > 0u);
        }
    };

    if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && uploadableArrayLayers > 0)
    {
        uint32_t layersToUploadMemorySize = eachLayerNeededMemory * uploadableArrayLayers;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + 0u;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + 0u;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + 0u;
        regionToCopyNext.imageExtent.width    = imageExtent.x;
        regionToCopyNext.imageExtent.height   = imageExtent.y;
        regionToCopyNext.imageExtent.depth    = imageExtent.z;
        regionToCopyNext.imageSubresource.layerCount = uploadableArrayLayers;

        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, layersToUploadMemorySize);

        bool copySuccess = performCopy(srcImageFormat, dstImageFormat, inCPUImage, outCPUImage, regionToCopyNext);

        if(copySuccess)
        {
            addToCurrentUploadBufferOffset(layersToUploadMemorySize);

            currentLayerInRegion += uploadableArrayLayers;
            updateCurrentOffsets();
            return true;
        }
        else
        {
            assert(false);
            return false;
        }
    }
    else if (currentBlockInRow == 0 && currentRowInSlice == 0 && canTransferMipLevelsPartially && uploadableSlices > 0)
    {
        // tryFillLayer();
        uint32_t slicesToUploadMemorySize = eachSliceNeededMemory * uploadableSlices;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + 0u;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + 0u;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
        regionToCopyNext.imageExtent.width    = imageExtent.x;
        regionToCopyNext.imageExtent.height   = imageExtent.y;
        regionToCopyNext.imageExtent.depth    = core::min(uploadableSlices * texelBlockDim.z, imageExtent.z);
        regionToCopyNext.imageSubresource.layerCount = 1u;
            
        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, slicesToUploadMemorySize);

        bool copySuccess = performCopy(srcImageFormat, dstImageFormat, inCPUImage, outCPUImage, regionToCopyNext);

        if(copySuccess)
        {
            addToCurrentUploadBufferOffset(slicesToUploadMemorySize);

            currentSliceInLayer += uploadableSlices;
            updateCurrentOffsets();
            return true;
        }
        else
        {
            assert(false);
            return false;
        }
    }
    else if (currentBlockInRow == 0 && canTransferMipLevelsPartially && uploadableRows > 0)
    {
        // tryFillSlice();
        uint32_t rowsToUploadMemorySize = eachRowNeededMemory * uploadableRows;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + 0u;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + currentRowInSlice * texelBlockDim.y;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
        regionToCopyNext.imageExtent.width    = imageExtent.x;
        regionToCopyNext.imageExtent.height   = core::min(uploadableRows * texelBlockDim.y, imageExtent.y);
        regionToCopyNext.imageExtent.depth    = core::min(1u * texelBlockDim.z, imageExtent.z);
        regionToCopyNext.imageSubresource.layerCount = 1u;
            
        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, rowsToUploadMemorySize);

        bool copySuccess = performCopy(srcImageFormat, dstImageFormat, inCPUImage, outCPUImage, regionToCopyNext);

        if(copySuccess)
        {
            addToCurrentUploadBufferOffset(rowsToUploadMemorySize);

            currentRowInSlice += uploadableRows;
            updateCurrentOffsets();
            return true;
        }
        else
        {
            assert(false);
            return false;
        }
    }
    else if (canTransferMipLevelsPartially && uploadableBlocks > 0)
    {
        // tryFillRow();
        uint32_t blocksToUploadMemorySize = eachBlockNeededMemory * uploadableBlocks;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + currentBlockInRow * texelBlockDim.x;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + currentRowInSlice * texelBlockDim.y;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
        regionToCopyNext.imageExtent.width    = core::min(uploadableBlocks * texelBlockDim.x, imageExtent.x);
        regionToCopyNext.imageExtent.height   = core::min(1u * texelBlockDim.y, imageExtent.y);
        regionToCopyNext.imageExtent.depth    = core::min(1u * texelBlockDim.z, imageExtent.z);
        regionToCopyNext.imageSubresource.layerCount = 1u;

        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, blocksToUploadMemorySize);

        bool copySuccess = performCopy(srcImageFormat, dstImageFormat, inCPUImage, outCPUImage, regionToCopyNext);

        if(copySuccess)
        {
            addToCurrentUploadBufferOffset(blocksToUploadMemorySize);

            currentBlockInRow += uploadableBlocks;
            updateCurrentOffsets();
            return true;
        }
        else
        {
            assert(false);
            return false;
        }
    }
    else
        return false;
}

} // namespace nbl::video