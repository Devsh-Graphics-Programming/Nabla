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

// Assumes no loss of precision due to block compression, only the endpoints
float getBcFormatMaxPrecision(asset::E_FORMAT format, uint32_t channel)
{
    if (channel == 3u)
    {
        switch (format)
        {
        // BC2 has 4 bit alpha
        case asset::EF_BC2_UNORM_BLOCK:
        case asset::EF_BC2_SRGB_BLOCK:
            return 1.f / 15.f;
        // BC3, BC7 and all ASTC formats have up to 8 bit alpha
        case asset::EF_BC3_UNORM_BLOCK:
        case asset::EF_BC3_SRGB_BLOCK:
        case asset::EF_BC7_UNORM_BLOCK:
        case asset::EF_BC7_SRGB_BLOCK:
        case asset::EF_ASTC_4x4_UNORM_BLOCK:
        case asset::EF_ASTC_4x4_SRGB_BLOCK:
        case asset::EF_ASTC_5x4_UNORM_BLOCK:
        case asset::EF_ASTC_5x4_SRGB_BLOCK:
        case asset::EF_ASTC_5x5_UNORM_BLOCK:
        case asset::EF_ASTC_5x5_SRGB_BLOCK:
        case asset::EF_ASTC_6x5_UNORM_BLOCK:
        case asset::EF_ASTC_6x5_SRGB_BLOCK:
        case asset::EF_ASTC_6x6_UNORM_BLOCK:
        case asset::EF_ASTC_6x6_SRGB_BLOCK:
        case asset::EF_ASTC_8x5_UNORM_BLOCK:
        case asset::EF_ASTC_8x5_SRGB_BLOCK:
        case asset::EF_ASTC_8x6_UNORM_BLOCK:
        case asset::EF_ASTC_8x6_SRGB_BLOCK:
        case asset::EF_ASTC_8x8_UNORM_BLOCK:
        case asset::EF_ASTC_8x8_SRGB_BLOCK:
        case asset::EF_ASTC_10x5_UNORM_BLOCK:
        case asset::EF_ASTC_10x5_SRGB_BLOCK:
        case asset::EF_ASTC_10x6_UNORM_BLOCK:
        case asset::EF_ASTC_10x6_SRGB_BLOCK:
        case asset::EF_ASTC_10x8_UNORM_BLOCK:
        case asset::EF_ASTC_10x8_SRGB_BLOCK:
        case asset::EF_ASTC_10x10_UNORM_BLOCK:
        case asset::EF_ASTC_10x10_SRGB_BLOCK:
        case asset::EF_ASTC_12x10_UNORM_BLOCK:
        case asset::EF_ASTC_12x10_SRGB_BLOCK:
        case asset::EF_ASTC_12x12_UNORM_BLOCK:
        case asset::EF_ASTC_12x12_SRGB_BLOCK:
            return 1.0 / 255.0;

        // Otherwise, assume binary (1 bit) alpha
        default:
            return 1.f;
        }
    }

    float rcpUnit = 0.0;
    switch (format)
    {
    case asset::EF_BC1_RGB_UNORM_BLOCK:
    case asset::EF_BC1_RGB_SRGB_BLOCK:
    case asset::EF_BC1_RGBA_UNORM_BLOCK:
    case asset::EF_BC1_RGBA_SRGB_BLOCK:
    case asset::EF_BC2_UNORM_BLOCK:
    case asset::EF_BC2_SRGB_BLOCK:
    case asset::EF_BC3_UNORM_BLOCK:
    case asset::EF_BC3_SRGB_BLOCK:
        rcpUnit = (channel == 1u) ? (1.0 / 63.0) : (1.0 / 31.0);
        break;
    case asset::EF_BC4_UNORM_BLOCK:
    case asset::EF_BC4_SNORM_BLOCK:
    case asset::EF_BC5_UNORM_BLOCK:
    case asset::EF_BC5_SNORM_BLOCK:
    case asset::EF_BC7_UNORM_BLOCK:
    case asset::EF_BC7_SRGB_BLOCK:
        rcpUnit = 1.0 / 255.0;
        break;
    case asset::EF_ASTC_4x4_UNORM_BLOCK:
    case asset::EF_ASTC_4x4_SRGB_BLOCK:
    case asset::EF_ASTC_5x4_UNORM_BLOCK:
    case asset::EF_ASTC_5x4_SRGB_BLOCK:
    case asset::EF_ASTC_5x5_UNORM_BLOCK:
    case asset::EF_ASTC_5x5_SRGB_BLOCK:
    case asset::EF_ASTC_6x5_UNORM_BLOCK:
    case asset::EF_ASTC_6x5_SRGB_BLOCK:
    case asset::EF_ASTC_6x6_UNORM_BLOCK:
    case asset::EF_ASTC_6x6_SRGB_BLOCK:
    case asset::EF_ASTC_8x5_UNORM_BLOCK:
    case asset::EF_ASTC_8x5_SRGB_BLOCK:
    case asset::EF_ASTC_8x6_UNORM_BLOCK:
    case asset::EF_ASTC_8x6_SRGB_BLOCK:
    case asset::EF_ASTC_8x8_UNORM_BLOCK:
    case asset::EF_ASTC_8x8_SRGB_BLOCK:
    case asset::EF_ASTC_10x5_UNORM_BLOCK:
    case asset::EF_ASTC_10x5_SRGB_BLOCK:
    case asset::EF_ASTC_10x6_UNORM_BLOCK:
    case asset::EF_ASTC_10x6_SRGB_BLOCK:
    case asset::EF_ASTC_10x8_UNORM_BLOCK:
    case asset::EF_ASTC_10x8_SRGB_BLOCK:
    case asset::EF_ASTC_10x10_UNORM_BLOCK:
    case asset::EF_ASTC_10x10_SRGB_BLOCK:
    case asset::EF_ASTC_12x10_UNORM_BLOCK:
    case asset::EF_ASTC_12x10_SRGB_BLOCK:
    case asset::EF_ASTC_12x12_UNORM_BLOCK:
    case asset::EF_ASTC_12x12_SRGB_BLOCK:
        // (All of these could be using HDR. Take extra flag to assume FP16 precision?)
        rcpUnit = 1.0 / 255.0;
        break;
    case asset::EF_EAC_R11_UNORM_BLOCK:
    case asset::EF_EAC_R11_SNORM_BLOCK:
    case asset::EF_EAC_R11G11_UNORM_BLOCK:
    case asset::EF_EAC_R11G11_SNORM_BLOCK:
        rcpUnit = 1.0 / 2047.0;
        break;
    case asset::EF_ETC2_R8G8B8_UNORM_BLOCK:
    case asset::EF_ETC2_R8G8B8_SRGB_BLOCK:
    case asset::EF_ETC2_R8G8B8A1_UNORM_BLOCK:
    case asset::EF_ETC2_R8G8B8A1_SRGB_BLOCK:
    case asset::EF_ETC2_R8G8B8A8_UNORM_BLOCK:
    case asset::EF_ETC2_R8G8B8A8_SRGB_BLOCK:
        rcpUnit = 1.0 / 31.0;
        break;
    case asset::EF_BC6H_UFLOAT_BLOCK:
    case asset::EF_BC6H_SFLOAT_BLOCK:
    {
        // BC6 isn't really FP16, so this is an over-estimation
        float f = 0.0;
        uint16_t f16 = core::Float16Compressor::compress(f);
        uint16_t dir = core::Float16Compressor::compress(2.f * (f + 1.f));
        return core::Float16Compressor::decompress(core::nextafter16(f16, dir)) - f;
    }
    case asset::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
        // TODO
        // I couldn't find any documentation on how this works
        return 0.0;
    }

    if (isSRGBFormat(format))
    {
        return core::srgb2lin(0.0 + rcpUnit) - core::srgb2lin(0.0);
    }

    return rcpUnit;
}

// Rules for promotion:
// - Cannot convert to block or planar format
// - Aspects: Preserved or added
// - Channel count: Preserved or increased
// - Data range: Preserved or increased (per channel)
// - Data precision: Preserved or improved (per channel)
//     - Bit depth when comparing non srgb
// If there are multiple matches: Pick smallest texel block
// srcFormat can't be in validFormats (no promotion should be made if the format itself is valid)
asset::E_FORMAT narrowDownFormatPromotion(core::unordered_set<asset::E_FORMAT> validFormats, asset::E_FORMAT srcFormat)
{
    asset::E_FORMAT smallestTexelBlock = asset::E_FORMAT::EF_UNKNOWN;
    uint32_t smallestTexelBlockSize = -1;

    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    assert(srcChannels);
    auto srcAspects = getImageAspects(srcFormat);
    bool srcBc = asset::isBlockCompressionFormat(srcFormat);

    float srcPrecision[4];
    float srcMinVal[4];
    float srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcPrecision[channel] = srcBc ? getBcFormatMaxPrecision(srcFormat, channel) : asset::getFormatPrecision(srcFormat, channel, 0.f);
        srcMinVal[channel] = asset::getFormatMinValue<float>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<float>(srcFormat, channel);
    }

    // Better way to iterate the bitflags here?
    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        if (!validFormats.contains(f))
            continue;

        // Can't transcode to BC or planar
        if (asset::isBlockCompressionFormat(f))
            continue;
        if (asset::isPlanarFormat(f))
            continue;

        // Can't have less channels
        auto dstChannels = asset::getFormatChannelCount(f);
        if (dstChannels < srcChannels)
            continue;

        {
            auto c = 0;
            while (c < srcChannels)
            {
                // Can't have less precision than source
                if (asset::getFormatPrecision(f, c, 0.f) < srcPrecision[c])
                    break;
                // Can't have less range than source
                if (asset::getFormatMinValue<float>(f, c) > srcMinVal[c] || asset::getFormatMaxValue<float>(f, c) < srcMaxVal[c])
                    break;

                c++;
            }
            // If we hit a break in the while loop
            if (c != srcChannels)
                continue;
        }

        // Can't have less aspects
        if ((getImageAspects(f) & srcAspects).value == srcAspects.value)
            continue;

        // Pick smallest valid format
        uint32_t texelBlockSize = asset::getTexelOrBlockBytesize(f);
        if (texelBlockSize < smallestTexelBlockSize)
        {
            smallestTexelBlockSize = texelBlockSize;
            smallestTexelBlock = f;
        }
    }

    return smallestTexelBlock;
}

asset::E_FORMAT IPhysicalDevice::promoteBufferFormat(const FormatPromotionRequest<asset::IBuffer::E_USAGE_FLAGS> req)
{
    auto& buf_cache = this->m_formatPromotionCache.buffers;
    auto cached = buf_cache.find(req);
    if (cached != buf_cache.end())
        return cached->second;

    video::IPhysicalDevice::SFormatBufferUsage srcUsages(req.usages);
    if (srcUsages < getBufferFormatUsages(req.originalFormat))
    {
        buf_cache.insert(cached, { req,req.originalFormat });
        return req.originalFormat;
    }

    // Cache valid formats per usage?
    core::unordered_set<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        // Checked earlier
        if (f == req.originalFormat)
            continue;

        if (srcUsages < getBufferFormatUsages(f))
        {
            validFormats.insert(f);
        }
    }

    auto promoted = narrowDownFormatPromotion(validFormats, req.originalFormat);
    buf_cache.insert(cached, { req,promoted });
    return promoted;
}

asset::E_FORMAT IPhysicalDevice::promoteImageFormat(const FormatPromotionRequest<asset::IImage::E_USAGE_FLAGS> req, const asset::IImage::E_TILING tiling)
{
    format_image_cache_t& cache = tiling == asset::IImage::E_TILING::ET_LINEAR 
        ? this->m_formatPromotionCache.linearTilingImages 
        : this->m_formatPromotionCache.optimalTilingImages;
    auto cached = cache.find(req);
    if (cached != cache.end())
        return cached->second;
    video::IPhysicalDevice::SFormatImageUsage srcUsages(req.usages);
    auto getImageFormatUsagesTiling = [&](asset::E_FORMAT f) {
        switch (tiling)
        {
        case asset::IImage::E_TILING::ET_LINEAR:
            return getImageFormatUsagesLinear(f);
        case asset::IImage::E_TILING::ET_OPTIMAL:
            return getImageFormatUsagesOptimal(f);
        default:
            assert(false); // Invalid tiling
        }
    };

    if (srcUsages < getImageFormatUsagesTiling(req.originalFormat))
    {
        cache.insert(cached, { req,req.originalFormat });
        return req.originalFormat;
    }

    // Cache valid formats per usage?
    core::unordered_set<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        // Checked earlier
        if (f == req.originalFormat)
            continue;

        if (srcUsages < getImageFormatUsagesTiling(f))
        {
            validFormats.insert(f);
        }
    }


    auto promoted = narrowDownFormatPromotion(validFormats, req.originalFormat);
    cache.insert(cached, { req,promoted });
    return promoted;
}

}