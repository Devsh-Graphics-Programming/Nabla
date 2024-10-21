#include "nbl/video/utilities/ImageRegionIterator.h"

namespace nbl::video
{

ImageRegionIterator::ImageRegionIterator(
    const std::span<const asset::IImage::SBufferCopy> copyRegions,
    IPhysicalDevice::SQueueFamilyProperties queueFamilyProps,
    const void* srcData,
    asset::E_FORMAT srcImageFormat,
    video::IGPUImage* const dstImage,
    size_t optimalRowPitchAlignment
)
    : regions(copyRegions)
    , minImageTransferGranularity(queueFamilyProps.minImageTransferGranularity)
    , srcImageFormat(srcImageFormat)
    , dstImageFormat(dstImage->getCreationParameters().format)
    , srcData(srcData)
    , dstImage(dstImage)
    , currentBlockInRow(0u)
    , currentRowInSlice(0u)
    , currentSliceInLayer(0u)
    , currentLayerInRegion(0u)
    , currentRegion(0u)
    , optimalRowPitchAlignment(optimalRowPitchAlignment)
{
    if(srcImageFormat == asset::EF_UNKNOWN)
        srcImageFormat = dstImageFormat;
    asset::TexelBlockInfo dstImageTexelBlockInfo(dstImageFormat);

    // bufferOffsetAlignment:
        // [x] If Depth/Stencil -> must be multiple of 4
        // [ ] If multi-planar -> bufferOffset must be a multiple of the element size of the compatible format for the aspectMask of imagesubresource
        // [x] If Queue doesn't support GRAPHICS_BIT or COMPUTE_BIT ->  must be multiple of 4
        // [x] bufferOffset must be a multiple of texel block size in bytes
    bufferOffsetAlignment = dstImageTexelBlockInfo.getBlockByteSize(); // can be non power of two
    if (asset::isDepthOrStencilFormat(dstImageFormat))
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);

    if (!(queueFamilyProps.queueFlags&(IQueue::FAMILY_FLAGS::COMPUTE_BIT|IQueue::FAMILY_FLAGS::GRAPHICS_BIT)))
        bufferOffsetAlignment = std::lcm(bufferOffsetAlignment, 4u);
    // TODO: Need to have a function to get equivalent format of the specific plane of this format (in aspectMask)
    // if(asset::isPlanarFormat(dstImageFormat->getCreationParameters().format))
        
    // Queues supporting graphics and/or compute operations must report (1,1,1) in minImageTransferGranularity, meaning that there are no additional restrictions on the granularity of image transfer operations for these queues.
    // Other queues supporting image transfer operations are only required to support whole mip level transfers, thus minImageTransferGranularity for queues belonging to such queue families may be (0,0,0)
    canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);

    auto dstImageParams = dstImage->getCreationParameters();

    /*
        We have to first construct two `ICPUImage`s per Region named `inCPUImage` and `outCPUImage`
        Then we will create fake ICPUBuffers that point to srcData and stagingBuffer with correct offsets
        Then we have to set the buffer and regions for each one of those ICPUImages using setBufferAndRegions
        Finally we fill the filter state and `execute` which require in/out CPUImages
    */

    imageFilterInCPUImages.resize(regions.size());
    imageFilterOutCPUImages.resize(regions.size());
    for (uint32_t i = 0; i < copyRegions.size(); ++i)
    {
        const auto region = regions[i];

        // inCPUImage is an image matching the params of dstImage but with the extents and layer count of the current region being copied and mipLevel 1u and the format being srcImageFormat
        auto& inCPUImage = imageFilterInCPUImages[i];

        asset::ICPUImage::SCreationParams inCPUImageParams = dstImageParams;
        inCPUImageParams.flags = asset::IImage::ECF_NONE; // Because we may want to write to first few layers of CUBEMAP (<6) but it's not valid to create an Cube ICPUImage with less that 6 layers.
        inCPUImageParams.format = srcImageFormat;
        inCPUImageParams.extent = region.imageExtent;
        inCPUImageParams.arrayLayers = region.imageSubresource.layerCount;
        inCPUImageParams.mipLevels = 1u; // since we copy one mip at a time to our dst image, it doesn't matter at the stage when we copy from cpu memory to staging memory
        inCPUImage = asset::ICPUImage::create(std::move(inCPUImageParams));
        assert(inCPUImage);
        
        auto inCPUImageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1);
        auto& inCpuImageRegion = inCPUImageRegions->front();
        inCpuImageRegion = {};
        inCpuImageRegion.bufferOffset = 0u;
        inCpuImageRegion.bufferRowLength = region.bufferRowLength;
        inCpuImageRegion.bufferImageHeight = region.bufferImageHeight;
        inCpuImageRegion.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
        inCpuImageRegion.imageSubresource.mipLevel = 0u;
        inCpuImageRegion.imageSubresource.baseArrayLayer = 0u;
        inCpuImageRegion.imageOffset.x = 0u;
        inCpuImageRegion.imageOffset.y = 0u;
        inCpuImageRegion.imageOffset.z = 0u;
        inCpuImageRegion.imageExtent.width = region.imageExtent.width;
        inCpuImageRegion.imageExtent.height = region.imageExtent.height;
        inCpuImageRegion.imageExtent.depth = region.imageExtent.depth;
        inCpuImageRegion.imageSubresource.layerCount = region.imageSubresource.layerCount;

        uint64_t offsetInCPUBuffer = region.bufferOffset;
        uint8_t* inCpuBufferPointer = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(srcData) + offsetInCPUBuffer);
        core::smart_refctd_ptr<asset::ICPUBuffer> inCPUBuffer = core::make_smart_refctd_ptr< asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(0xdeadbeefBADC0FFEull, inCpuBufferPointer, core::adopt_memory);
        inCPUImage->setBufferAndRegions(std::move(inCPUBuffer), inCPUImageRegions);
        assert(inCPUImage->getBuffer());
        assert(inCPUImage->getRegions().size() > 0u);


        // outCPUImage is an image matching the params of dstImage but with the extents and layer count of the current region being copied and mipLevel 1u
        auto& outCPUImage = imageFilterOutCPUImages[i];
        outCPUImageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1);
        
        asset::ICPUImage::SCreationParams outCPUImageParams = dstImageParams;
        outCPUImageParams.flags = asset::IImage::ECF_NONE; // Because we may want to write to first few layers of CUBEMAP (<6) but it's not valid to create an Cube ICPUImage with less that 6 layers.
        outCPUImageParams.extent = region.imageExtent;
        outCPUImageParams.arrayLayers = region.imageSubresource.layerCount;
        outCPUImageParams.mipLevels = 1u;
        outCPUImage = asset::ICPUImage::create(std::move(outCPUImageParams));
        assert(outCPUImage);
    }
}

size_t ImageRegionIterator::getMemoryNeededForRemainingRegions() const
{
    asset::TexelBlockInfo dstImageTexelBlockInfo(dstImageFormat);
    assert(dstImageTexelBlockInfo.getBlockByteSize() > 0u);
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

        auto imageExtentInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth));

        const auto copyTexelStrides = getOptimalCopyTexelStrides(region.imageExtent);
        const core::vector4du32_SIMD copyByteStrides = dstImageTexelBlockInfo.convert3DTexelStridesTo1DByteStrides(copyTexelStrides);

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

// This Swizzles makes sure copying from srcFormat image to promotedFormat image is consistent and extra "unused" channels will be ZERO and alpha will be ONE
template<unsigned int SRC_CHANNELS>
struct PromotionComponentSwizzle
{
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
    const core::vector4du32_SIMD& inOffsetBaseLayer,
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
    state.inOffsetBaseLayer = inOffsetBaseLayer;
    state.outOffsetBaseLayer = core::vectorSIMDu32(0u);
    state.inMipLevel = 0u;
    state.outMipLevel = 0u;

    if (filter.execute(core::execution::par_unseq, &state))
        return true;
    else
        return false;
}

bool performIntermediateCopy(
    asset::E_FORMAT srcImageFormat,
    asset::E_FORMAT dstImageFormat,
    const core::vector4du32_SIMD& inOffsetBaseLayer,
    const core::smart_refctd_ptr<asset::ICPUImage>& inCPUImage,
    const core::smart_refctd_ptr<asset::ICPUImage>& outCPUImage,
    const asset::IImage::SBufferCopy& region)
{
    // In = srcData, Out = stagingBuffer
    if (srcImageFormat == dstImageFormat)
    {
        return performCopyUsingImageFilter<asset::CCopyImageFilter>(inOffsetBaseLayer, inCPUImage, outCPUImage, region);
    }
    else
    {
        auto srcChannelCount = asset::getFormatChannelCount(srcImageFormat);
        if (srcChannelCount == 1u)
            return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<1u>>>(inOffsetBaseLayer, inCPUImage, outCPUImage, region);
        else if (srcChannelCount == 2u)
            return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<2u>>>(inOffsetBaseLayer, inCPUImage, outCPUImage, region);
        else if (srcChannelCount == 3u)
            return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<3u>>>(inOffsetBaseLayer, inCPUImage, outCPUImage, region);
        else
            return performCopyUsingImageFilter<asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN, PromotionComponentSwizzle<4u>>>(inOffsetBaseLayer, inCPUImage, outCPUImage, region);
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

    const asset::TexelBlockInfo dstImageTexelBlockInfo(dstImageFormat);

    // ! Current Region that may break down into smaller regions (the first smaller region is nextRegionToCopy)
    const asset::IImage::SBufferCopy & mainRegion = regions[currentRegion];

    // ! We only need subresourceSize for validations and assertions about minImageTransferGranularity because granularity requirements can be ignored if region fits against the right corner of the subresource (described in more detail below)
    const auto subresourceSize = dstImage->getMipSize(mainRegion.imageSubresource.mipLevel);
    const auto subresourceSizeInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(subresourceSize);

    const auto texelBlockDim = dstImageTexelBlockInfo.getDimension();

    // ! Don't confuse imageExtent with subresourceSize, imageExtent is the extent of the main region to copy and the subresourceSize is the actual size of dstImage 
    const auto imageOffsetInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(mainRegion.imageOffset.x, mainRegion.imageOffset.y, mainRegion.imageOffset.z));
    const auto imageExtentInBlocks = dstImageTexelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(mainRegion.imageExtent.width, mainRegion.imageExtent.height, mainRegion.imageExtent.depth));

    const auto copyBlockStrides = dstImageTexelBlockInfo.convertTexelsToBlocks(getOptimalCopyTexelStrides(mainRegion.imageExtent));
    const core::vector4du32_SIMD copyByteStrides = dstImageTexelBlockInfo.convert3DBlockStridesTo1DByteStrides(copyBlockStrides);

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
        // Cached because we can
        inCPUImage = imageFilterInCPUImages[currentRegion];
        outCPUImage = imageFilterOutCPUImages[currentRegion];

        // But we need to set outCPUImage regions and buffer since that cannot be known at initialization time.
        auto& outCpuImageRegion = outCPUImageRegions->front();
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

        uint8_t* outCpuBufferPointer = reinterpret_cast<uint8_t*>(stagingBufferPointer) + stagingBufferOffset;
        core::smart_refctd_ptr<asset::ICPUBuffer> outCPUBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(outCPUBufferSize, outCpuBufferPointer, core::adopt_memory);
        outCPUImage->setBufferAndRegions(std::move(outCPUBuffer), outCPUImageRegions);
        assert(outCPUImage->getBuffer());
        assert(outCPUImage->getRegions().size() > 0u);
    };

    if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && uploadableArrayLayers > 0)
    {
        uint32_t layersToUploadMemorySize = eachLayerNeededMemory * uploadableArrayLayers;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = copyBlockStrides.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = copyBlockStrides.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + 0u;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + 0u;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + 0u;
        regionToCopyNext.imageExtent.width    = mainRegion.imageExtent.width;
        regionToCopyNext.imageExtent.height   = mainRegion.imageExtent.height;
        regionToCopyNext.imageExtent.depth    = mainRegion.imageExtent.depth;
        regionToCopyNext.imageSubresource.layerCount = uploadableArrayLayers;

        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, layersToUploadMemorySize);

        const auto inOffsetBaseLayer = core::vector4du32_SIMD(currentBlockInRow * texelBlockDim.x, currentRowInSlice * texelBlockDim.y, currentSliceInLayer * texelBlockDim.z, currentLayerInRegion);
        bool copySuccess = performIntermediateCopy(srcImageFormat, dstImageFormat, inOffsetBaseLayer, inCPUImage, outCPUImage, regionToCopyNext);

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
        uint32_t slicesToUploadMemorySize = eachSliceNeededMemory * uploadableSlices;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = copyBlockStrides.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = copyBlockStrides.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + 0u;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + 0u;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
        regionToCopyNext.imageExtent.width    = mainRegion.imageExtent.width;
        regionToCopyNext.imageExtent.height   = mainRegion.imageExtent.height;
        regionToCopyNext.imageExtent.depth    = core::min(uploadableSlices * texelBlockDim.z, mainRegion.imageExtent.depth);
        regionToCopyNext.imageSubresource.layerCount = 1u;
            
        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, slicesToUploadMemorySize);

        const auto inOffsetBaseLayer = core::vector4du32_SIMD(currentBlockInRow * texelBlockDim.x, currentRowInSlice * texelBlockDim.y, currentSliceInLayer * texelBlockDim.z, currentLayerInRegion);
        bool copySuccess = performIntermediateCopy(srcImageFormat, dstImageFormat, inOffsetBaseLayer, inCPUImage, outCPUImage, regionToCopyNext);

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
        uint32_t rowsToUploadMemorySize = eachRowNeededMemory * uploadableRows;

        regionToCopyNext.bufferOffset = stagingBufferOffset;
        regionToCopyNext.bufferRowLength = copyBlockStrides.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = copyBlockStrides.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + 0u;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + currentRowInSlice * texelBlockDim.y;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
        regionToCopyNext.imageExtent.width    = mainRegion.imageExtent.width;
        regionToCopyNext.imageExtent.height   = core::min(uploadableRows * texelBlockDim.y, mainRegion.imageExtent.height);
        regionToCopyNext.imageExtent.depth    = core::min(1u * texelBlockDim.z, mainRegion.imageExtent.depth);
        regionToCopyNext.imageSubresource.layerCount = 1u;
            
        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, rowsToUploadMemorySize);

        const auto inOffsetBaseLayer = core::vector4du32_SIMD(currentBlockInRow * texelBlockDim.x, currentRowInSlice * texelBlockDim.y, currentSliceInLayer * texelBlockDim.z, currentLayerInRegion);
        bool copySuccess = performIntermediateCopy(srcImageFormat, dstImageFormat, inOffsetBaseLayer, inCPUImage, outCPUImage, regionToCopyNext);

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
        regionToCopyNext.bufferRowLength = copyBlockStrides.x * texelBlockDim.x;
        regionToCopyNext.bufferImageHeight = copyBlockStrides.y * texelBlockDim.y;
        regionToCopyNext.imageSubresource.aspectMask = mainRegion.imageSubresource.aspectMask;
        regionToCopyNext.imageSubresource.mipLevel = mainRegion.imageSubresource.mipLevel;
        regionToCopyNext.imageSubresource.baseArrayLayer = mainRegion.imageSubresource.baseArrayLayer + currentLayerInRegion;
        regionToCopyNext.imageOffset.x = mainRegion.imageOffset.x + currentBlockInRow * texelBlockDim.x;
        regionToCopyNext.imageOffset.y = mainRegion.imageOffset.y + currentRowInSlice * texelBlockDim.y;
        regionToCopyNext.imageOffset.z = mainRegion.imageOffset.z + currentSliceInLayer * texelBlockDim.z;
        regionToCopyNext.imageExtent.width    = core::min(uploadableBlocks * texelBlockDim.x, mainRegion.imageExtent.width);
        regionToCopyNext.imageExtent.height   = core::min(1u * texelBlockDim.y, mainRegion.imageExtent.height);
        regionToCopyNext.imageExtent.depth    = core::min(1u * texelBlockDim.z, mainRegion.imageExtent.depth);
        regionToCopyNext.imageSubresource.layerCount = 1u;

        core::smart_refctd_ptr<asset::ICPUImage> inCPUImage;
        core::smart_refctd_ptr<asset::ICPUImage> outCPUImage;
        createMockInOutCPUImagesForFilter(inCPUImage, outCPUImage, blocksToUploadMemorySize);

        const auto inOffsetBaseLayer = core::vector4du32_SIMD(currentBlockInRow * texelBlockDim.x, currentRowInSlice * texelBlockDim.y, currentSliceInLayer * texelBlockDim.z, currentLayerInRegion);
        bool copySuccess = performIntermediateCopy(srcImageFormat, dstImageFormat, inOffsetBaseLayer, inCPUImage, outCPUImage, regionToCopyNext);

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