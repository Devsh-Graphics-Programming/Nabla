#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

IDebugCallback* IPhysicalDevice::getDebugCallback() const
{
    return m_initData.api->getDebugCallback();
}

bool IPhysicalDevice::validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const
{
    for (auto i=0u; i<m_initData.qfamProperties->size(); i++)
    {
        const auto& qci = params.queueParams[i];

        const auto& qfam = m_initData.qfamProperties->operator[](i);
        if (qci.count>qfam.queueCount)
            return false;

        for (uint32_t i=0u; i<qci.count; ++i)
        {
            const float priority = qci.priorities[i];
            if (priority<0.f || priority>1.f)
                return false;
        }
    }
    
    if(!params.featuresToEnable.isSubsetOf(m_initData.features))
        return false; // Requested features are not all supported by physical device

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
        // The color channels for BC1, BC2 & BC3 are RGB565
        rcpUnit = (channel == 1u) ? (1.0 / 63.0) : (1.0 / 31.0);
        // Weights also allow for more precision. These formats have 2 bit weights
        rcpUnit *= 1.0 / 3.0;
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
        return core::Float16Compressor::decompress(1) - 0.0;
    }
    case asset::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
        // TODO: Use proper metrics here instead of assuming full 8 bit
        return 1.0 / 255.0;
    }

    if (isSRGBFormat(format))
    {
        return core::srgb2lin(0.0 + rcpUnit) - core::srgb2lin(0.0);
    }

    return rcpUnit;
}

double getFormatPrecisionAt(asset::E_FORMAT format, uint32_t channel, double value)
{
    if (asset::isBlockCompressionFormat(format))
        return getBcFormatMaxPrecision(format, channel);
    switch (format)
    {
    case asset::EF_E5B9G9R9_UFLOAT_PACK32:
    {
        // Minimum precision value would be a 9bit mantissa & 5bit exponent float
        // (This ignores the shared exponent)
        int bitshft = 2;

        uint16_t f16 = core::Float16Compressor::compress(value);
        uint16_t enc = f16 >> bitshft;
        uint16_t next_f16 = (enc + 1) << bitshft;

        return core::Float16Compressor::decompress(next_f16) - value;
    }
    default: return asset::getFormatPrecision(format, channel, value);
    }
}

// Returns true if 'a' is not equal to 'b' and can be promoted FROM 'b'
bool canPromoteFormat(asset::E_FORMAT a, asset::E_FORMAT b, bool srcSignedFormat, bool srcIntFormat, uint32_t srcChannels, double srcMin[], double srcMax[])
{
    // The value itself should already have been checked to not be valid before calling this
    if (a == b)
        return false;
    // Can't transcode to BC or planar
    if (asset::isBlockCompressionFormat(a))
        return false;
    if (asset::isPlanarFormat(a))
        return false;
    // Can't promote between int and normalized/float/scaled formats
    if (asset::isIntegerFormat(a) != srcIntFormat)
        return false;
    // Can't promote between signed and unsigned formats in integers
    // (this causes a different sampler type to be necessary in the shader)
    if (srcIntFormat && asset::isSignedFormat(a) != srcSignedFormat)
        return false;
    // Can't have less channels
    if (asset::getFormatChannelCount(a) < srcChannels)
        return false;

    // Can't have less precision or value range
    for (uint32_t c = 0; c < srcChannels; c++)
    {
        double mina = asset::getFormatMinValue<double>(a, c),
            minb = asset::getFormatMinValue<double>(b, c),
            maxa = asset::getFormatMaxValue<double>(a, c),
            maxb = asset::getFormatMaxValue<double>(b, c);

        // return false if a has less precision (higher precision delta) than b
        // check at 0, since precision is non-increasing
        // also check at min & max, since there's potential for cross-over with constant formats
        if (getFormatPrecisionAt(a, c, 0.0) > getFormatPrecisionAt(b, c, 0.0)
                || getFormatPrecisionAt(a, c, srcMin[c]) > getFormatPrecisionAt(b, c, srcMin[c])
                || getFormatPrecisionAt(a, c, srcMax[c]) > getFormatPrecisionAt(b, c, srcMax[c]))
            return false;
        // return false if a has less range than b
        if (mina > minb || maxa < maxb)
            return false;
    }
    return true;
}

double getFormatPrecisionMaxDt(asset::E_FORMAT f, uint32_t c, double srcMin, double srcMax)
{
    return std::max(std::max(getFormatPrecisionAt(f, c, 0.0), getFormatPrecisionAt(f, c, srcMin)), getFormatPrecisionAt(f, c, srcMax));
}

// Returns true if 'a' is a better fit than 'b' (for tie breaking)
// Tie-breaking rules:
// - RGBA vs BGRA matches srcFormat
// - Maximum precision delta is smaller
// - Value range is larger
bool isFormatBetterFit(asset::E_FORMAT a, asset::E_FORMAT b, bool srcBgra, uint32_t srcChannels, double srcMin[], double srcMax[])
{
    assert(asset::getTexelOrBlockBytesize(a) == asset::getTexelOrBlockBytesize(b));
    bool curBgraMatch = asset::isBGRALayoutFormat(a) == srcBgra;
    bool prevBgraMatch = asset::isBGRALayoutFormat(b) == srcBgra;

    // if one of the two fits the original bgra better, use that
    if (curBgraMatch != prevBgraMatch)
        return curBgraMatch;

    // Check precision deltas
    double precisionDeltasA[4];
    double precisionDeltasB[4];
    for (uint32_t c = 0; c < srcChannels; c++)
    {
        // View comments above about value selection
        // Pick the max precision delta for each format
        precisionDeltasA[c] = getFormatPrecisionMaxDt(a, c, srcMin[c], srcMax[c]);
        precisionDeltasB[c] = getFormatPrecisionMaxDt(b, c, srcMin[c], srcMax[c]);

        // if one of the two has a better max precision delta, use that
        if (precisionDeltasA[c] != precisionDeltasB[c])
            return precisionDeltasA[c] < precisionDeltasB[c];
    }

    // Check difference in quantifiable values within the ranges for a and b
    double wasteA = 0.0;
    double wasteB = 0.0;
    for (uint32_t c = 0; c < srcChannels; c++)
    {
        double mina = asset::getFormatMinValue<double>(a, c),
            minb = asset::getFormatMinValue<double>(b, c),
            maxa = asset::getFormatMaxValue<double>(a, c),
            maxb = asset::getFormatMaxValue<double>(b, c);
        assert(mina <= srcMin[c] && maxa >= srcMax[c] &&
            minb <= srcMin[c] && maxb >= srcMax[c]);

        wasteA += (srcMin[c] - mina) / precisionDeltasA[c];
        wasteA += (maxa - srcMax[c]) / precisionDeltasA[c];

        wasteB += (srcMin[c] - minb) / precisionDeltasB[c];
        wasteB += (maxb - srcMax[c]) / precisionDeltasB[c];
    }

    // if one of the two has less "waste" of quantifiable values, use that
    if (wasteA != wasteB)
        return wasteA < wasteB;

    return false;
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
asset::E_FORMAT narrowDownFormatPromotion(const core::unordered_set<asset::E_FORMAT>& validFormats, asset::E_FORMAT srcFormat)
{
    if (validFormats.empty()) return asset::EF_UNKNOWN;

    asset::E_FORMAT smallestTexelBlock = asset::EF_UNKNOWN;
    uint32_t smallestTexelBlockSize = -1;

    bool srcBgra = asset::isBGRALayoutFormat(srcFormat);
    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    double srcMinVal[4];
    double srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcMinVal[channel] = asset::getFormatMinValue<double>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<double>(srcFormat, channel);
    }

    for (auto iter = validFormats.begin(); iter != validFormats.end(); iter++)
    {
        asset::E_FORMAT f = *iter;

        uint32_t texelBlockSize = asset::getTexelOrBlockBytesize(f);
        // Don't promote if we have a better valid format already
        if (texelBlockSize > smallestTexelBlockSize) {
            continue;
        }

        if (texelBlockSize == smallestTexelBlockSize)
        {
            if (!isFormatBetterFit(f, smallestTexelBlock, srcBgra, srcChannels, srcMinVal, srcMaxVal))
                continue;
        }

        smallestTexelBlockSize = texelBlockSize;
        smallestTexelBlock = f;
    }

    assert(smallestTexelBlock != asset::EF_UNKNOWN);
    return smallestTexelBlock;
}

asset::E_FORMAT IPhysicalDevice::promoteBufferFormat(const SBufferFormatPromotionRequest req) const
{
    assert(
        !asset::isBlockCompressionFormat(req.originalFormat) &&
        !asset::isPlanarFormat(req.originalFormat) &&
        getImageAspects(req.originalFormat).hasFlags(asset::IImage::EAF_COLOR_BIT)
    );
    auto& buf_cache = this->m_formatPromotionCache.buffers;
    auto cached = buf_cache.find(req);
    if (cached != buf_cache.end())
        return cached->second;

    // don't need to promote
    if ((req.usages&getBufferFormatUsages()[req.originalFormat])==req.usages)
    {
        buf_cache.insert(cached, { req,req.originalFormat });
        return req.originalFormat;
    }

    auto srcFormat = req.originalFormat;
    bool srcIntFormat = asset::isIntegerFormat(srcFormat);
    bool srcSignedFormat = asset::isSignedFormat(srcFormat);
    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    double srcMinVal[4];
    double srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcMinVal[channel] = asset::getFormatMinValue<double>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<double>(srcFormat, channel);
    }

    // Cache valid formats per usage?
    core::unordered_set<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        // Can't have less aspects
        if (!getImageAspects(f).hasFlags(asset::IImage::EAF_COLOR_BIT))
            continue;

        if (!canPromoteFormat(f, srcFormat, srcSignedFormat, srcIntFormat, srcChannels, srcMinVal, srcMaxVal))
            continue;

        if ((req.usages&getBufferFormatUsages()[f])==req.usages)
            validFormats.insert(f);
    }

    auto promoted = narrowDownFormatPromotion(validFormats, req.originalFormat);
    buf_cache.insert(cached, { req,promoted });
    return promoted;
}

asset::E_FORMAT IPhysicalDevice::promoteImageFormat(const SImageFormatPromotionRequest req, const IGPUImage::TILING tiling) const
{
    format_image_cache_t& cache = tiling==IGPUImage::TILING::LINEAR 
        ? this->m_formatPromotionCache.linearTilingImages 
        : this->m_formatPromotionCache.optimalTilingImages;
    auto cached = cache.find(req);
    if (cached != cache.end())
        return cached->second;

    // don't need to promote
    if ((req.usages&getImageFormatUsages(tiling)[req.originalFormat])==req.usages)
    {
        cache.insert(cached, { req,req.originalFormat });
        return req.originalFormat;
    }

    auto srcFormat = req.originalFormat;
    auto srcAspects = getImageAspects(srcFormat);
    bool srcIntFormat = asset::isIntegerFormat(srcFormat);
    bool srcSignedFormat = asset::isSignedFormat(srcFormat);
    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    double srcMinVal[4];
    double srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcMinVal[channel] = asset::getFormatMinValue<double>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<double>(srcFormat, channel);
    }

    // Cache valid formats per usage?
    core::unordered_set<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        // Can't have less aspects
        if (!getImageAspects(f).hasFlags(srcAspects))
            continue;

        if (!canPromoteFormat(f, srcFormat, srcSignedFormat, srcIntFormat, srcChannels, srcMinVal, srcMaxVal))
            continue;

        if ((req.usages&getImageFormatUsages(tiling)[f])==req.usages)
            validFormats.insert(f);
    }


    auto promoted = narrowDownFormatPromotion(validFormats, req.originalFormat);
    cache.insert(cached, { req,promoted });
    return promoted;
}

}
