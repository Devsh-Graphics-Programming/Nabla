#include "nbl/video/utilities/ImageRegionIterator.h"

namespace nbl::video
{

ImageRegionIterator::ImageRegionIterator(
    const std::span<const asset::IImage::SBufferCopy> copyRegions,
    IPhysicalDevice::SQueueFamilyProperties queueFamilyProps,
    const video::IGPUImage* const image,
    size_t optimalRowPitchAlignment
)
    : regions(copyRegions)
    , minImageTransferGranularity(queueFamilyProps.minImageTransferGranularity)
    , imageFormat(image->getCreationParameters().format)
    , image(image)
    , currentBlockInRow(0u)
    , currentRowInSlice(0u)
    , currentSliceInLayer(0u)
    , currentLayerInRegion(0u)
    , currentRegion(0u)
    , optimalRowPitchAlignment(optimalRowPitchAlignment)
{
    asset::TexelBlockInfo texelBlockInfo(imageFormat);

    // bufferOffsetAlignment, same rules for `copyBufferToImage` and `copyImageToBuffer`, see the VUIDs in the header:
        // [x] If Depth/Stencil -> must be multiple of 4
        // [ ] If multi-planar -> bufferOffset must be a multiple of the element size of the compatible format for the aspectMask of imagesubresource
        // [x] If Queue doesn't support GRAPHICS_BIT or COMPUTE_BIT ->  must be multiple of 4
        // [x] bufferOffset must be a multiple of texel block size in bytes
    bufferOffsetAlignment = texelBlockInfo.getBlockByteSize(); // can be non power of two
    if (asset::isDepthOrStencilFormat(imageFormat))
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);

    if (!(queueFamilyProps.queueFlags&(IQueue::FAMILY_FLAGS::COMPUTE_BIT|IQueue::FAMILY_FLAGS::GRAPHICS_BIT)))
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);
    // TODO: Need to have a function to get equivalent format of the specific plane of this format (in aspectMask)
    // if(asset::isPlanarFormat(imageFormat))

    // Queues supporting graphics and/or compute operations must report (1,1,1) in minImageTransferGranularity, meaning that there are no additional restrictions on the granularity of image transfer operations for these queues.
    // Other queues supporting image transfer operations are only required to support whole mip level transfers, thus minImageTransferGranularity for queues belonging to such queue families may be (0,0,0)
    canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);
}

size_t ImageRegionIterator::getMemoryNeededForRemainingRegions() const
{
    asset::TexelBlockInfo texelBlockInfo(imageFormat);
    assert(texelBlockInfo.getBlockByteSize() > 0u);
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

        auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth));

        const auto copyTexelStrides = getOptimalCopyTexelStrides(region.imageExtent);
        const core::vector4du32_SIMD copyByteStrides = texelBlockInfo.convert3DTexelStridesTo1DByteStrides(copyTexelStrides);

        if (i == currentRegion)
        {
            auto remainingBlocksInRow = imageExtentInBlocks.x - currentBlockInRow;
            auto remainingRowsInSlice = imageExtentInBlocks.y - currentRowInSlice;
            auto remainingSlicesInLayer = imageExtentInBlocks.z - currentSliceInLayer;
            auto remainingLayersInRegion = region.imageSubresource.layerCount - currentLayerInRegion;

            if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && remainingLayersInRegion > 0)
            {
                incrementMemoryNeeded(copyByteStrides[3] * remainingLayersInRegion);
            }
            else if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer > 0)
            {
                incrementMemoryNeeded(copyByteStrides[2] * remainingSlicesInLayer);
                if (remainingLayersInRegion > 1u)
                    incrementMemoryNeeded(copyByteStrides[3] * (remainingLayersInRegion - 1u));
            }
            else if (currentBlockInRow == 0 && currentRowInSlice > 0)
            {
                incrementMemoryNeeded(copyByteStrides[1] * remainingRowsInSlice);

                if (remainingSlicesInLayer > 1u)
                    incrementMemoryNeeded(copyByteStrides[2] * (remainingSlicesInLayer - 1u));
                if (remainingLayersInRegion > 1u)
                    incrementMemoryNeeded(copyByteStrides[3] * (remainingLayersInRegion - 1u));
            }
            else if (currentBlockInRow > 0)
            {
                // want to first fill the remaining blocks in current row
                incrementMemoryNeeded(copyByteStrides[0] * remainingBlocksInRow);
                // then fill the remaining rows in current slice
                if (remainingRowsInSlice > 1u)
                    incrementMemoryNeeded(copyByteStrides[1] * (remainingRowsInSlice - 1u));
                // then fill the remaining slices in current layer
                if (remainingSlicesInLayer > 1u)
                    incrementMemoryNeeded(copyByteStrides[2] * (remainingSlicesInLayer - 1u));
                // then fill the remaining layers in current region
                if (remainingLayersInRegion > 1u)
                    incrementMemoryNeeded(copyByteStrides[3] * (remainingLayersInRegion - 1u));
            }
        }
        else
        {
            // we want to fill the whole layers in the region
            incrementMemoryNeeded(copyByteStrides[3] * region.imageSubresource.layerCount); // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z * region.imageSubresource.layerCount
        }
    }
    return memoryNeededForRemainingRegions;
}

bool ImageRegionIterator::advance(SChunk& chunk, uint32_t& availableMemory, uint32_t& stagingBufferOffset)
{
    if(isFinished())
        return false;

    auto addToStagingBufferOffset = [&](uint32_t size) -> bool
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
    if(!addToStagingBufferOffset(0u))
    {
        return false;
    }

    const asset::TexelBlockInfo texelBlockInfo(imageFormat);

    // ! Current Region that may break down into smaller regions (the first smaller region is the chunk)
    const asset::IImage::SBufferCopy & mainRegion = regions[currentRegion];

    // ! We only need subresourceSize for validations and assertions about minImageTransferGranularity because granularity requirements can be ignored if region fits against the right corner of the subresource (described in more detail below)
    const auto subresourceSize = image->getMipSize(mainRegion.imageSubresource.mipLevel);
    const auto subresourceSizeInBlocks = texelBlockInfo.convertTexelsToBlocks(subresourceSize);

    const auto texelBlockDim = texelBlockInfo.getDimension();

    // ! Don't confuse imageExtent with subresourceSize, imageExtent is the extent of the main region to copy and the subresourceSize is the actual size of the image
    const auto imageOffsetInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(mainRegion.imageOffset.x, mainRegion.imageOffset.y, mainRegion.imageOffset.z));
    const auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(mainRegion.imageExtent.width, mainRegion.imageExtent.height, mainRegion.imageExtent.depth));

    const auto copyBlockStrides = texelBlockInfo.convertTexelsToBlocks(getOptimalCopyTexelStrides(mainRegion.imageExtent));
    const core::vector4du32_SIMD copyByteStrides = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(copyBlockStrides);

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

    uint32_t eachBlockNeededMemory  = copyByteStrides[0];  // = blockByteSize
    uint32_t eachRowNeededMemory    = copyByteStrides[1];  // = blockByteSize * copyBlockStrides.x
    uint32_t eachSliceNeededMemory  = copyByteStrides[2];  // = blockByteSize * copyBlockStrides.x * copyBlockStrides.y
    uint32_t eachLayerNeededMemory  = copyByteStrides[3];  // = blockByteSize * copyBlockStrides.x * copyBlockStrides.y * copyBlockStrides.z

    // There is remaining layers in region that needs copying
    uint32_t transferableArrayLayers = availableMemory / eachLayerNeededMemory;
    uint32_t remainingLayers = mainRegion.imageSubresource.layerCount - currentLayerInRegion;
    transferableArrayLayers = core::min(transferableArrayLayers, remainingLayers);
    // A: There is remaining layers left in region -> Copy Slices (Depths)
    uint32_t transferableSlices = availableMemory / eachSliceNeededMemory;
    uint32_t remainingSlices = imageExtentInBlocks.z - currentSliceInLayer;
    transferableSlices = core::min(transferableSlices, remainingSlices);
    if(transferableSlices > 0 && minImageTransferGranularity.depth > 1u && (imageOffsetInBlocks.z + currentSliceInLayer + transferableSlices) < subresourceSizeInBlocks.z)
        transferableSlices = core::alignDown(transferableSlices, minImageTransferGranularity.depth);
    // B: There is remaining slices left in layer -> Copy Rows
    uint32_t transferableRows = availableMemory / eachRowNeededMemory;
    uint32_t remainingRows = imageExtentInBlocks.y - currentRowInSlice;
    transferableRows = core::min(transferableRows, remainingRows);
    if(transferableRows > 0 && minImageTransferGranularity.height > 1u && (imageOffsetInBlocks.y + currentRowInSlice + transferableRows) < subresourceSizeInBlocks.y)
        transferableRows = core::alignDown(transferableRows, minImageTransferGranularity.height);
    // C: There is remaining slices left in layer -> Copy Blocks
    uint32_t transferableBlocks = availableMemory / eachBlockNeededMemory;
    uint32_t remainingBlocks = imageExtentInBlocks.x - currentBlockInRow;
    transferableBlocks = core::min(transferableBlocks, remainingBlocks);
    if(transferableBlocks > 0 && minImageTransferGranularity.width > 1u && (imageOffsetInBlocks.x + currentBlockInRow + transferableBlocks) < subresourceSizeInBlocks.x)
        transferableBlocks = core::alignDown(transferableBlocks, minImageTransferGranularity.width);

    auto& staging = chunk.staging;
    staging = {};
    staging.bufferOffset = stagingBufferOffset;
    staging.bufferRowLength = copyBlockStrides.x * texelBlockDim.x;
    staging.bufferImageHeight = copyBlockStrides.y * texelBlockDim.y;
    staging.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
    staging.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
    staging.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
    staging.imageSubresource.layerCount = 1u;
    staging.imageOffset.x = mainRegion.imageOffset.x + currentBlockInRow * texelBlockDim.x;
    staging.imageOffset.y = mainRegion.imageOffset.y + currentRowInSlice * texelBlockDim.y;
    staging.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
    staging.imageExtent.width    = mainRegion.imageExtent.width;
    staging.imageExtent.height   = mainRegion.imageExtent.height;
    staging.imageExtent.depth    = mainRegion.imageExtent.depth;

    // VUID-vkCmdCopyBufferToImage-dstImage-00207/00208/00209, VUID-vkCmdCopyImageToBuffer-srcImage-00207/00208/00209
    const uint32_t remainingWidth = mainRegion.imageExtent.width - currentBlockInRow * texelBlockDim.x;
    const uint32_t remainingHeight = mainRegion.imageExtent.height - currentRowInSlice * texelBlockDim.y;
    const uint32_t remainingDepth = mainRegion.imageExtent.depth - currentSliceInLayer * texelBlockDim.z;

    uint32_t chunkByteSize = 0u;
    uint32_t layerStep = 0u, sliceStep = 0u, rowStep = 0u, blockStep = 0u;
    if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && transferableArrayLayers > 0)
    {
        chunkByteSize = eachLayerNeededMemory * transferableArrayLayers;
        staging.imageSubresource.layerCount = transferableArrayLayers;
        layerStep = transferableArrayLayers;
    }
    else if (currentBlockInRow == 0 && currentRowInSlice == 0 && canTransferMipLevelsPartially && transferableSlices > 0)
    {
        chunkByteSize = eachSliceNeededMemory * transferableSlices;
        staging.imageExtent.depth = core::min(transferableSlices * texelBlockDim.z, remainingDepth);
        sliceStep = transferableSlices;
    }
    else if (currentBlockInRow == 0 && canTransferMipLevelsPartially && transferableRows > 0)
    {
        chunkByteSize = eachRowNeededMemory * transferableRows;
        staging.imageExtent.height = core::min(transferableRows * texelBlockDim.y, remainingHeight);
        staging.imageExtent.depth = core::min(1u * texelBlockDim.z, remainingDepth);
        rowStep = transferableRows;
    }
    else if (canTransferMipLevelsPartially && transferableBlocks > 0)
    {
        chunkByteSize = eachBlockNeededMemory * transferableBlocks;
        staging.imageExtent.width = core::min(transferableBlocks * texelBlockDim.x, remainingWidth);
        staging.imageExtent.height = core::min(1u * texelBlockDim.y, remainingHeight);
        staging.imageExtent.depth = core::min(1u * texelBlockDim.z, remainingDepth);
        blockStep = transferableBlocks;
    }
    else
        return false;

    chunk.stagingByteSize = chunkByteSize;
    chunk.imageFormat = imageFormat;
    chunk.hostBufferOffset = mainRegion.bufferOffset;
    chunk.hostTexelStrides = mainRegion.getTexelStrides();
    chunk.hostOffsetBaseLayer = core::vector4du32_SIMD(currentBlockInRow * texelBlockDim.x, currentRowInSlice * texelBlockDim.y, currentSliceInLayer * texelBlockDim.z, currentLayerInRegion);

    addToStagingBufferOffset(chunkByteSize);

    currentLayerInRegion += layerStep;
    currentSliceInLayer += sliceStep;
    currentRowInSlice += rowStep;
    currentBlockInRow += blockStep;
    updateCurrentOffsets();
    return true;
}

constexpr asset::ICPUImageView::SComponentMapping getPromotionComponentMapping(const uint32_t srcChannels)
{
    return {
        asset::ICPUImageView::SComponentMapping::ES_R,
        srcChannels > 1u ? asset::ICPUImageView::SComponentMapping::ES_G:asset::ICPUImageView::SComponentMapping::ES_ZERO,
        srcChannels > 2u ? asset::ICPUImageView::SComponentMapping::ES_B:asset::ICPUImageView::SComponentMapping::ES_ZERO,
        srcChannels > 3u ? asset::ICPUImageView::SComponentMapping::ES_A:asset::ICPUImageView::SComponentMapping::ES_ONE
    };
}

// This Swizzles makes sure copying from srcFormat image to promotedFormat image is consistent and extra "unused" channels will be ZERO and alpha will be ONE
template<unsigned int SRC_CHANNELS>
struct PromotionComponentSwizzle
{
    static constexpr asset::ICPUImageView::SComponentMapping componentMapping = getPromotionComponentMapping(SRC_CHANNELS);

    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

        reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];

        if constexpr (SRC_CHANNELS > 1)
            reinterpret_cast<out_t*>(out)[1u] = reinterpret_cast<const in_t*>(in)[1u];
        else
            reinterpret_cast<out_t*>(out)[1u] = static_cast<in_t>(0);

        if constexpr (SRC_CHANNELS > 2)
            reinterpret_cast<out_t*>(out)[2u] = reinterpret_cast<const in_t*>(in)[2u];
        else
            reinterpret_cast<out_t*>(out)[2u] = static_cast<in_t>(0);

        if constexpr (SRC_CHANNELS > 3)
            reinterpret_cast<out_t*>(out)[3u] = reinterpret_cast<const in_t*>(in)[3u];
        else
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

bool performIntermediateCopy(
    asset::E_FORMAT inFormat,
    const core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage,
    const core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage,
    const asset::IImage::SBufferCopy& region)
{
    auto inChannelCount = asset::getFormatChannelCount(inFormat);
    if (inChannelCount == 1u)
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<1u>>>(inCPUImage, outCPUImage, region);
    else if (inChannelCount == 2u)
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<2u>>>(inCPUImage, outCPUImage, region);
    else if (inChannelCount == 3u)
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<3u>>>(inCPUImage, outCPUImage, region);
    else
        return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<4u>>>(inCPUImage, outCPUImage, region);
}

bool copyChunkFiltered(const ImageRegionIterator::SChunk& chunk, const bool toStaging, uint8_t* const stagingChunk, uint8_t* const hostChunk, const asset::E_FORMAT hostFormat)
{
    const auto& staging = chunk.staging;
    const uint32_t layerCount = core::max(staging.imageSubresource.layerCount, 1u);

    auto createMockImage = [&](const asset::E_FORMAT format, const uint32_t bufferRowLength, const uint32_t bufferImageHeight, void* data, const size_t dataSize) -> core::smart_refctd_ptr<asset::ICPUImage>
    {
        asset::ICPUImage::SCreationParams params = {};
        params.type = staging.imageExtent.depth > 1u ? asset::IImage::E_TYPE::ET_3D:asset::IImage::E_TYPE::ET_2D;
        params.samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
        params.format = format;
        params.extent = staging.imageExtent;
        params.mipLevels = 1u;
        params.arrayLayers = layerCount;
        params.flags = asset::IImage::ECF_NONE;
        auto image = asset::ICPUImage::create(std::move(params));
        if (!image)
            return nullptr;

        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1);
        auto& region = regions->front();
        region = {};
        region.bufferOffset = 0ull;
        region.bufferRowLength = bufferRowLength;
        region.bufferImageHeight = bufferImageHeight;
        region.imageSubresource.aspectMask = staging.imageSubresource.aspectMask;
        region.imageSubresource.mipLevel = 0u;
        region.imageSubresource.baseArrayLayer = 0u;
        region.imageSubresource.layerCount = layerCount;
        region.imageOffset.x = 0;
        region.imageOffset.y = 0;
        region.imageOffset.z = 0;
        region.imageExtent = staging.imageExtent;

        auto buffer = asset::ICPUBuffer::create({ { dataSize }, data, core::getNullMemoryResource() }, core::adopt_memory);
        if (!buffer || !image->setBufferAndRegions(std::move(buffer), regions))
            return nullptr;
        return image;
    };

    const auto hostRowLengthAndImageHeight = asset::TexelBlockInfo(hostFormat).roundToBlockSize(chunk.hostTexelStrides);
    auto hostImage = createMockImage(hostFormat, hostRowLengthAndImageHeight.x, hostRowLengthAndImageHeight.y, hostChunk, 0xdeadbeefBADC0FFEull);
    auto stagingImage = createMockImage(chunk.imageFormat, staging.bufferRowLength, staging.bufferImageHeight, stagingChunk, chunk.stagingByteSize);
    if (!hostImage || !stagingImage)
        return false;

    return performIntermediateCopy(
        toStaging ? hostFormat:chunk.imageFormat,
        toStaging ? hostImage:stagingImage,
        toStaging ? stagingImage:hostImage,
        staging
    );
}

bool ImageRegionIterator::copyChunk(const SChunk& chunk, const E_DIRECTION direction, void* stagingPtr, void* hostPtr, const asset::E_FORMAT hostFormat)
{
    const asset::TexelBlockInfo hostBlockInfo(hostFormat);
    const size_t hostBlockByteSize = hostBlockInfo.getBlockByteSize();
    const auto hostBlockStrides = hostBlockInfo.convertTexelsToBlocks(chunk.hostTexelStrides);
    const auto hostOffsetInBlocks = hostBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(chunk.hostOffsetBaseLayer.x, chunk.hostOffsetBaseLayer.y, chunk.hostOffsetBaseLayer.z));
    const size_t hostRowPitch = hostBlockStrides.x * hostBlockByteSize;
    const size_t hostSlicePitch = hostRowPitch * hostBlockStrides.y;
    const size_t hostLayerPitch = hostSlicePitch * hostBlockStrides.z;

    uint8_t* const hostChunk = reinterpret_cast<uint8_t*>(hostPtr) + chunk.hostBufferOffset
        + chunk.hostOffsetBaseLayer.w * hostLayerPitch + hostOffsetInBlocks.z * hostSlicePitch + hostOffsetInBlocks.y * hostRowPitch + hostOffsetInBlocks.x * hostBlockByteSize;
    uint8_t* const stagingChunk = reinterpret_cast<uint8_t*>(stagingPtr) + chunk.staging.bufferOffset;

    const bool toStaging = direction == E_DIRECTION::HOST_TO_STAGING;
    const asset::TexelBlockInfo imageBlockInfo(chunk.imageFormat);
    const size_t imageBlockByteSize = imageBlockInfo.getBlockByteSize();
    auto extentInBlocks = imageBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(chunk.staging.imageExtent.width, chunk.staging.imageExtent.height, chunk.staging.imageExtent.depth));
    extentInBlocks.w = core::max(chunk.staging.imageSubresource.layerCount, 1u);
    const auto stagingBlockStrides = chunk.staging.getBlockStrides(imageBlockInfo);
    const size_t stagingRowPitch = stagingBlockStrides.x * imageBlockByteSize;
    const size_t stagingSlicePitch = stagingRowPitch * stagingBlockStrides.y;
    assert(size_t(extentInBlocks.w - 1u) * stagingSlicePitch * stagingBlockStrides.z + size_t(extentInBlocks.z - 1u) * stagingSlicePitch + size_t(extentInBlocks.y - 1u) * stagingRowPitch + size_t(extentInBlocks.x) * imageBlockByteSize <= chunk.stagingByteSize);
    if (hostFormat == chunk.imageFormat)
    {
        if (toStaging)
            asset::CBasicImageFilterCommon::copyBlocks(core::execution::par_unseq, stagingChunk, stagingBlockStrides, hostChunk, hostBlockStrides, imageBlockByteSize, extentInBlocks);
        else
            asset::CBasicImageFilterCommon::copyBlocks(core::execution::par_unseq, hostChunk, hostBlockStrides, stagingChunk, stagingBlockStrides, imageBlockByteSize, extentInBlocks);
        return true;
    }

    const asset::E_FORMAT inFormat = toStaging ? hostFormat:chunk.imageFormat;
    if (asset::STexelReshuffle reshuffle; asset::STexelReshuffle::create(inFormat, toStaging ? chunk.imageFormat:hostFormat, getPromotionComponentMapping(asset::getFormatChannelCount(inFormat)), reshuffle))
    {
        if (toStaging)
            reshuffle.execute(core::execution::par_unseq, stagingChunk, stagingBlockStrides, hostChunk, hostBlockStrides, extentInBlocks);
        else
            reshuffle.execute(core::execution::par_unseq, hostChunk, hostBlockStrides, stagingChunk, stagingBlockStrides, extentInBlocks);
        return true;
    }
    return copyChunkFiltered(chunk, toStaging, stagingChunk, hostChunk, hostFormat);
}

} // namespace nbl::video
