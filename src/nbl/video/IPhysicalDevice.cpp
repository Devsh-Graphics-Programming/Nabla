#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

IPhysicalDevice::IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc) :
    m_system(std::move(s)), m_GLSLCompiler(std::move(glslc))
{
    // TODO(Erfan): Add defualt values for these and remove memsets
    memset(&m_properties, 0, sizeof(SProperties));
    memset(&m_features, 0, sizeof(SFeatures));
    memset(&m_memoryProperties, 0, sizeof(SMemoryProperties));
    memset(&m_linearTilingUsages, 0, sizeof(SFormatImageUsage));
    memset(&m_optimalTilingUsages, 0, sizeof(SFormatImageUsage));
    memset(&m_bufferUsages, 0, sizeof(SFormatBufferUsage));
}

void IPhysicalDevice::addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderdoc)
{
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_UBO_SIZE",m_properties.limits.maxUBOSize);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_SSBO_SIZE",m_properties.limits.maxSSBOSize);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_BUFFER_VIEW_TEXELS",m_properties.limits.maxBufferViewTexels);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_BUFFER_SIZE",core::min(m_properties.limits.maxBufferSize, ~0u));
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_IMAGE_ARRAY_LAYERS",m_properties.limits.maxImageArrayLayers);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_PER_STAGE_SSBO_COUNT",m_properties.limits.maxPerStageDescriptorSSBOs);
    
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_SSBO_COUNT",m_properties.limits.maxDescriptorSetSSBOs);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_UBO_COUNT",m_properties.limits.maxDescriptorSetUBOs);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_TEXTURE_COUNT",m_properties.limits.maxDescriptorSetImages);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_STORAGE_IMAGE_COUNT",m_properties.limits.maxDescriptorSetStorageImages);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_DRAW_INDIRECT_COUNT",m_properties.limits.maxDrawIndirectCount);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MIN_POINT_SIZE",m_properties.limits.pointSizeRange[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_POINT_SIZE",m_properties.limits.pointSizeRange[1]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MIN_LINE_WIDTH",m_properties.limits.lineWidthRange[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_LINE_WIDTH",m_properties.limits.lineWidthRange[1]);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_VIEWPORTS",m_properties.limits.maxViewports);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_VIEWPORT_WIDTH",m_properties.limits.maxViewportDims[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_VIEWPORT_HEIGHT",m_properties.limits.maxViewportDims[1]);

    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_WORKGROUP_SIZE_X",m_properties.limits.maxWorkgroupSize[0]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_WORKGROUP_SIZE_Y",m_properties.limits.maxWorkgroupSize[1]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_WORKGROUP_SIZE_Z",m_properties.limits.maxWorkgroupSize[2]);
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS",m_properties.limits.maxOptimallyResidentWorkgroupInvocations);

    // TODO: Need upper and lower bounds on workgroup sizes!
    // TODO: Need to know if subgroup size is constant/known
    addGLSLDefineToPool(pool,"NBL_LIMIT_SUBGROUP_SIZE",m_properties.limits.subgroupSize);
    // TODO: @achal test examples 14 and 48 on all APIs and GPUs
    
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_RESIDENT_INVOCATIONS",m_properties.limits.maxResidentInvocations);


    // TODO: Add feature defines


    if (runningInRenderdoc)
        addGLSLDefineToPool(pool,"NBL_RUNNING_IN_RENDERDOC");
}

bool IPhysicalDevice::validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const
{
    using range_t = core::SRange<const ILogicalDevice::SQueueCreationParams>;
    range_t qcis(params.queueParams, params.queueParams+params.queueParamsCount);

    for (const auto& qci : qcis)
    {
        if (qci.familyIndex >= m_qfamProperties->size())
            return false;

        const auto& qfam = (*m_qfamProperties)[qci.familyIndex];
        if (qci.count == 0u)
            return false;
        if (qci.count > qfam.queueCount)
            return false;

        for (uint32_t i = 0u; i < qci.count; ++i)
        {
            const float priority = qci.priorities[i];
            if (priority < 0.f)
                return false;
            if (priority > 1.f)
                return false;
        }
    }

    return true;
}

inline core::bitflag<asset::IImage::E_ASPECT_FLAGS> getImageAspects(asset::E_FORMAT _fmt)
{
    core::bitflag<asset::IImage::E_ASPECT_FLAGS> flags;
    bool depthOrStencil = asset::isDepthOrStencilFormat(_fmt);
    bool stencilOnly = asset::isStencilOnlyFormat(_fmt);
    bool depthOnly = asset::isDepthOnlyFormat(_fmt);
    if (depthOrStencil || depthOnly) flags |= asset::IImage::EAF_DEPTH_BIT;
    if (depthOrStencil || stencilOnly) flags |= asset::IImage::EAF_STENCIL_BIT;
    if (!depthOrStencil && !stencilOnly && !depthOnly) flags |= asset::IImage::EAF_COLOR_BIT;

    return flags;
}

// Rules for promotion:
// - Cannot convert to block format
// - Aspects: Preserved or added
// - Channel count: Preserved or increased
// - Data range: Preserved or increased (per channel)
// - Data precision: Preserved or improved (per channel)
//     - Bit depth when comparing non srgb
// If there are multiple matches: Pick smallest texel block
asset::E_FORMAT narrowDownFormatPromotion(core::bitflag<asset::E_FORMAT> validFormats, asset::E_FORMAT srcFormat)
{
    asset::E_FORMAT smallestTexelBlock = asset::E_FORMAT::EF_UNKNOWN;
    uint32_t smallestTexelBlockSize = -1;

    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    if (!srcChannels) return asset::EF_UNKNOWN;
    auto srcAspects = getImageAspects(srcFormat);
    auto srcBitDepth = asset::getMaxChannelBitDepth(srcFormat);
    // TODO "Magic" value of 128.0 until I figure out a value-independent way of comparing precision
    auto srcPrecision = asset::getFormatPrecision(srcFormat, srcChannels, 128.0);

    // Better way to iterate the bitflags here?
    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        if (!validFormats.hasFlags(f))
        {
            continue;
        }

        auto dstChannels = asset::getFormatChannelCount(f);
        // Verify if promotion is valid from srcFormat -> format
        bool promotionValid;
        promotionValid = asset::isBlockCompressionFormat(f) ? f == srcFormat : true; // Can't transcode to compressed formats
        promotionValid = promotionValid && dstChannels >= srcChannels; // Channel count
        promotionValid = promotionValid && (getImageAspects(f) & srcAspects).value == srcAspects.value; // Aspects
        promotionValid = promotionValid && asset::getMaxChannelBitDepth(f) >= srcBitDepth; // Data range
        promotionValid = promotionValid && asset::getFormatPrecision(f, dstChannels, 128.0) >= srcPrecision; // Precision

        uint32_t texelBlockSize = getTexelOrBlockBytesize(f);
        if (promotionValid && texelBlockSize < smallestTexelBlockSize)
        {
            smallestTexelBlockSize = texelBlockSize;
            smallestTexelBlock = f;
        }
    }

    return smallestTexelBlock;
}

video::IPhysicalDevice::SFormatBufferUsage convBufferUsage(core::bitflag<asset::IBuffer::E_USAGE_FLAGS> usages)
{
    video::IPhysicalDevice::SFormatBufferUsage formatBufUsg;
    formatBufUsg.isInitialized = 1;
    formatBufUsg.vertexAttribute = usages.hasFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
    formatBufUsg.bufferView = usages.hasFlags(asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT);
    formatBufUsg.storageBufferView = usages.hasFlags(asset::IBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT);
    // Special flags for these?
    formatBufUsg.accelerationStructureVertex = usages.hasFlags(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
    formatBufUsg.storageBufferViewAtomic = usages.hasFlags(asset::IBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT);

    return formatBufUsg;
}

asset::E_FORMAT IPhysicalDevice::promoteBufferFormat(const FormatPromotionRequest<asset::IBuffer::E_USAGE_FLAGS> req)
{
    auto buf_cache = this->m_formatPromotionCache.buffers;
    auto cached = buf_cache.find(req);
    if (cached != buf_cache.end())
        return cached->second;

    auto srcUsages = convBufferUsage(req.usages);

    // Cache valid formats per usage?
    core::bitflag<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        auto formatUsages = this->getBufferFormatUsages(f);
        if ((srcUsages & formatUsages) == srcUsages)
        {
            validFormats |= f;
        }
    }

    //if (validFormats.hasFlags(req.originalFormat)) return req.originalFormat;

    return narrowDownFormatPromotion(validFormats, req.originalFormat);
}

video::IPhysicalDevice::SFormatImageUsage convImageUsage(core::bitflag<asset::IImage::E_USAGE_FLAGS> usages)
{
    video::IPhysicalDevice::SFormatImageUsage formatImgUsg;
    formatImgUsg.isInitialized = 1;
    formatImgUsg.sampledImage = usages.hasFlags(asset::IImage::EUF_SAMPLED_BIT);
    formatImgUsg.storageImage = usages.hasFlags(asset::IImage::EUF_STORAGE_BIT);
    formatImgUsg.transferSrc = usages.hasFlags(asset::IImage::EUF_TRANSFER_SRC_BIT);
    formatImgUsg.transferDst = usages.hasFlags(asset::IImage::EUF_TRANSFER_DST_BIT);
    formatImgUsg.attachment = (usages & core::bitflag<asset::IImage::E_USAGE_FLAGS>(
        asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT)).value != 0;
    // Special flags for these?
    formatImgUsg.blitSrc = usages.hasFlags(asset::IImage::EUF_TRANSFER_SRC_BIT);
    formatImgUsg.blitDst = usages.hasFlags(asset::IImage::EUF_TRANSFER_DST_BIT);
    formatImgUsg.storageImageAtomic = usages.hasFlags(asset::IImage::EUF_STORAGE_BIT);
    formatImgUsg.attachmentBlend = (usages & core::bitflag<asset::IImage::E_USAGE_FLAGS>(
        asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT)).value != 0;

    return formatImgUsg;
}

asset::E_FORMAT IPhysicalDevice::promoteImageFormat(const FormatPromotionRequest<asset::IImage::E_USAGE_FLAGS> req, const asset::IImage::E_TILING tiling)
{
    format_image_cache_t cache;
    switch (tiling)
    {
    case asset::IImage::E_TILING::ET_LINEAR:
        cache = this->m_formatPromotionCache.linearTilingImages;
        break;
    case asset::IImage::E_TILING::ET_OPTIMAL:
        cache = this->m_formatPromotionCache.optimalTilingImages;
        break;
    default:
        return asset::E_FORMAT::EF_UNKNOWN;
    }
    auto cached = cache.find(req);
    if (cached != cache.end())
        return cached->second;
    auto srcUsages = convImageUsage(req.usages);

    // Cache valid formats per usage?
    core::bitflag<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        video::IPhysicalDevice::SFormatImageUsage formatUsages;
        switch (tiling)
        {
        case asset::IImage::E_TILING::ET_LINEAR:
            formatUsages = this->getImageFormatUsagesLinear(f);
            break;
        case asset::IImage::E_TILING::ET_OPTIMAL:
            formatUsages = this->getImageFormatUsagesOptimal(f);
            break;
        default:
            return asset::E_FORMAT::EF_UNKNOWN;
        }
        auto commonUsages = srcUsages & formatUsages;
        commonUsages.log2MaxSamples = srcUsages.log2MaxSamples; // TODO: Handle sample counts
        if (commonUsages == srcUsages)
        {
            validFormats |= f;
        }
    }

    //if (validFormats.hasFlags(req.originalFormat)) return req.originalFormat;

    return narrowDownFormatPromotion(validFormats, req.originalFormat);
}

}