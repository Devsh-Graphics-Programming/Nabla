#ifndef __IRR_CONVERT_COLOR_H_INCLUDED__
#define __IRR_CONVERT_COLOR_H_INCLUDED__

#include "EColorFormat.h"
#include "decodePixels.h"
#include "encodePixels.h"

namespace irr { namespace video
{
    namespace impl
    {
        template<bool SCALED, ECOLOR_FORMAT cf, typename T>
        struct SCallDecode;

        template<ECOLOR_FORMAT cf, typename T>
        struct SCallDecode<false, cf, T>
        {
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY);
            }
        };
        template<ECOLOR_FORMAT cf, typename T>
        struct SCallDecode<true, cf, T>
        {
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY, _scale);
            }
        };

        template<bool SCALED, ECOLOR_FORMAT cf, typename T>
        struct SCallEncode;

        template<ECOLOR_FORMAT cf, typename T>
        struct SCallEncode<false, cf, T>
        {
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                encodePixels<cf, T>(_pix, _input);
            }
        };
        template<ECOLOR_FORMAT cf, typename T>
        struct SCallEncode<true, cf, T>
        {
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                encodePixels<cf, T>(_pix, _input, _scale);
            }
        };
    } //namespace impl


    template<ECOLOR_FORMAT sF, ECOLOR_FORMAT dF>
    inline void convertColor(const void* srcPix[4], void* dstPix, uint64_t _scale, uint32_t _blockX, uint32_t _blockY)
    {
        if (isIntegerFormat<sF>() && isIntegerFormat<dF>())
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            //encodePixels<dF, encT>(dstPix, reinterpret_cast<encT*>(decbuf), _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, reinterpret_cast<encT*>(decbuf), _scale);
        }
        else if (
            (isNormalizedFormat<sF>() && isNormalizedFormat<dF>()) ||
            (isFloatingPointFormat<sF>() && isFloatingPointFormat<dF>()) ||
            (isNormalizedFormat<sF>() && isFloatingPointFormat<dF>()) ||
            (isFloatingPointFormat<sF>() && isNormalizedFormat<dF>())
        )
        {
            using decT = double;
            using encT = double;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            //encodePixels<dF, encT>(dstPix, decbuf, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, decbuf, _scale);
        }
        else if ((isFloatingPointFormat<sF>() || isNormalizedFormat<sF>()) && isIntegerFormat<dF>())
        {
            using decT = double;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            //encodePixels<dF, encT>(dstPix, encbuf, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, encbuf, _scale);
        }
        else if (isIntegerFormat<sF>() && (isNormalizedFormat<dF>() || isFloatingPointFormat<dF>()))
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = double;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            //encodePixels<dF, encT>(dstPix, encbuf, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, encbuf, _scale);
        }
    }
    template<ECOLOR_FORMAT sF, ECOLOR_FORMAT dF>
    inline void convertColor(const void* srcPix[4], void* dstPix, uint64_t _scale, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize)
    {
        const uint32_t srcStride = getTexelOrBlockSize(sF);
        const uint32_t dstStride = getTexelOrBlockSize(dF);

        uint32_t hPlaneReduction[4], vPlaneReduction[4], chCntInPlane[4];
        getHorizontalReductionFactorPerPlane(sF, hPlaneReduction);
        getVerticalReductionFactorPerPlane(sF, vPlaneReduction);
        getChannelsPerPlane(sF, chCntInPlane);

        const core::vector3d<uint32_t> sdims = getBlockDimensions(sF);

        const uint8_t** src = reinterpret_cast<const uint8_t**>(srcPix);
        uint8_t* const dst_begin = reinterpret_cast<uint8_t*>(dstPix);
        for (size_t i = 0u; i < _pixOrBlockCnt; ++i)
        {
            // assuming _imgSize is always represented in texels
            const uint32_t px = i % (_imgSize.X / sdims.X);
            const uint32_t py = i / (_imgSize.X / sdims.X);
            //px, py are block or texel position
            //x, y are position within block
            for (uint32_t x = 0u; x < sdims.X; ++x)
            {
                for (uint32_t y = 0u; y < sdims.Y; ++y)
                    convertColor<sF, dF>(reinterpret_cast<const void**>(src), dst_begin + ((sdims.Y * py + y)*_imgSize.X + px*sdims.X + x), _scale, x, y);
            }
            if (!isPlanarFormat<sF>())
            {
                src[0] += srcStride;
            }
            else
            {
                const uint32_t px = i % _imgSize.X;
                const uint32_t py = i / _imgSize.X;
                for (uint32_t j = 0u; j < 4u; ++j)
                    src[j] = reinterpret_cast<const uint8_t*>(srcPix[j]) + chCntInPlane[j]*((_imgSize.X/hPlaneReduction[j]) * (py/vPlaneReduction[j]) + px/hPlaneReduction[j]);
            }
        }
    }

    namespace impl 
    {
    template<ECOLOR_FORMAT sF>
    inline void convertColor_RTimpl(ECOLOR_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, uint64_t _scale, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize)
    {
        switch (_dfmt)
        {
        case ECF_A1R5G5B5: return convertColor<sF, ECF_A1R5G5B5>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R5G6B5: return convertColor<sF, ECF_R5G6B5>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R4G4_UNORM_PACK8: return convertColor<sF, ECF_R4G4_UNORM_PACK8>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R4G4B4A4_UNORM_PACK16: return convertColor<sF, ECF_R4G4B4A4_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B4G4R4A4_UNORM_PACK16: return convertColor<sF, ECF_B4G4R4A4_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R5G6B5_UNORM_PACK16: return convertColor<sF, ECF_R5G6B5_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B5G6R5_UNORM_PACK16: return convertColor<sF, ECF_B5G6R5_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R5G5B5A1_UNORM_PACK16: return convertColor<sF, ECF_R5G5B5A1_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B5G5R5A1_UNORM_PACK16: return convertColor<sF, ECF_B5G5R5A1_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A1R5G5B5_UNORM_PACK16: return convertColor<sF, ECF_A1R5G5B5_UNORM_PACK16>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_UNORM: return convertColor<sF, ECF_R8_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SNORM: return convertColor<sF, ECF_R8_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_USCALED: return convertColor<sF, ECF_R8_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SSCALED: return convertColor<sF, ECF_R8_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_UINT: return convertColor<sF, ECF_R8_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SINT: return convertColor<sF, ECF_R8_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SRGB: return convertColor<sF, ECF_R8_SRGB>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_UNORM: return convertColor<sF, ECF_R8G8_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SNORM: return convertColor<sF, ECF_R8G8_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_USCALED: return convertColor<sF, ECF_R8G8_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SSCALED: return convertColor<sF, ECF_R8G8_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_UINT: return convertColor<sF, ECF_R8G8_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SINT: return convertColor<sF, ECF_R8G8_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SRGB: return convertColor<sF, ECF_R8G8_SRGB>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_UNORM: return convertColor<sF, ECF_R8G8B8_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SNORM: return convertColor<sF, ECF_R8G8B8_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_USCALED: return convertColor<sF, ECF_R8G8B8_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SSCALED: return convertColor<sF, ECF_R8G8B8_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_UINT: return convertColor<sF, ECF_R8G8B8_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SINT: return convertColor<sF, ECF_R8G8B8_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SRGB: return convertColor<sF, ECF_R8G8B8_SRGB>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_UNORM: return convertColor<sF, ECF_B8G8R8_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SNORM: return convertColor<sF, ECF_B8G8R8_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_USCALED: return convertColor<sF, ECF_B8G8R8_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SSCALED: return convertColor<sF, ECF_B8G8R8_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_UINT: return convertColor<sF, ECF_B8G8R8_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SINT: return convertColor<sF, ECF_B8G8R8_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SRGB: return convertColor<sF, ECF_B8G8R8_SRGB>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_UNORM: return convertColor<sF, ECF_R8G8B8A8_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SNORM: return convertColor<sF, ECF_R8G8B8A8_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_USCALED: return convertColor<sF, ECF_R8G8B8A8_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SSCALED: return convertColor<sF, ECF_R8G8B8A8_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_UINT: return convertColor<sF, ECF_R8G8B8A8_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SINT: return convertColor<sF, ECF_R8G8B8A8_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SRGB: return convertColor<sF, ECF_R8G8B8A8_SRGB>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_UNORM: return convertColor<sF, ECF_B8G8R8A8_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SNORM: return convertColor<sF, ECF_B8G8R8A8_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_USCALED: return convertColor<sF, ECF_B8G8R8A8_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SSCALED: return convertColor<sF, ECF_B8G8R8A8_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_UINT: return convertColor<sF, ECF_B8G8R8A8_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SINT: return convertColor<sF, ECF_B8G8R8A8_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SRGB: return convertColor<sF, ECF_B8G8R8A8_SRGB>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_UNORM_PACK32: return convertColor<sF, ECF_A8B8G8R8_UNORM_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SNORM_PACK32: return convertColor<sF, ECF_A8B8G8R8_SNORM_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_USCALED_PACK32: return convertColor<sF, ECF_A8B8G8R8_USCALED_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SSCALED_PACK32: return convertColor<sF, ECF_A8B8G8R8_SSCALED_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_UINT_PACK32: return convertColor<sF, ECF_A8B8G8R8_UINT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SINT_PACK32: return convertColor<sF, ECF_A8B8G8R8_SINT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SRGB_PACK32: return convertColor<sF, ECF_A8B8G8R8_SRGB_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_UNORM_PACK32: return convertColor<sF, ECF_A2R10G10B10_UNORM_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_SNORM_PACK32: return convertColor<sF, ECF_A2R10G10B10_SNORM_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_USCALED_PACK32: return convertColor<sF, ECF_A2R10G10B10_USCALED_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_SSCALED_PACK32: return convertColor<sF, ECF_A2R10G10B10_SSCALED_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_UINT_PACK32: return convertColor<sF, ECF_A2R10G10B10_UINT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_SINT_PACK32: return convertColor<sF, ECF_A2R10G10B10_SINT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_UNORM_PACK32: return convertColor<sF, ECF_A2B10G10R10_UNORM_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_SNORM_PACK32: return convertColor<sF, ECF_A2B10G10R10_SNORM_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_USCALED_PACK32: return convertColor<sF, ECF_A2B10G10R10_USCALED_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_SSCALED_PACK32: return convertColor<sF, ECF_A2B10G10R10_SSCALED_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_UINT_PACK32: return convertColor<sF, ECF_A2B10G10R10_UINT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_SINT_PACK32: return convertColor<sF, ECF_A2B10G10R10_SINT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_UNORM: return convertColor<sF, ECF_R16_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SNORM: return convertColor<sF, ECF_R16_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_USCALED: return convertColor<sF, ECF_R16_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SSCALED: return convertColor<sF, ECF_R16_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_UINT: return convertColor<sF, ECF_R16_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SINT: return convertColor<sF, ECF_R16_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SFLOAT: return convertColor<sF, ECF_R16_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_UNORM: return convertColor<sF, ECF_R16G16_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SNORM: return convertColor<sF, ECF_R16G16_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_USCALED: return convertColor<sF, ECF_R16G16_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SSCALED: return convertColor<sF, ECF_R16G16_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_UINT: return convertColor<sF, ECF_R16G16_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SINT: return convertColor<sF, ECF_R16G16_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SFLOAT: return convertColor<sF, ECF_R16G16_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_UNORM: return convertColor<sF, ECF_R16G16B16_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SNORM: return convertColor<sF, ECF_R16G16B16_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_USCALED: return convertColor<sF, ECF_R16G16B16_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SSCALED: return convertColor<sF, ECF_R16G16B16_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_UINT: return convertColor<sF, ECF_R16G16B16_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SINT: return convertColor<sF, ECF_R16G16B16_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SFLOAT: return convertColor<sF, ECF_R16G16B16_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_UNORM: return convertColor<sF, ECF_R16G16B16A16_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SNORM: return convertColor<sF, ECF_R16G16B16A16_SNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_USCALED: return convertColor<sF, ECF_R16G16B16A16_USCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SSCALED: return convertColor<sF, ECF_R16G16B16A16_SSCALED>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_UINT: return convertColor<sF, ECF_R16G16B16A16_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SINT: return convertColor<sF, ECF_R16G16B16A16_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SFLOAT: return convertColor<sF, ECF_R16G16B16A16_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32_UINT: return convertColor<sF, ECF_R32_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32_SINT: return convertColor<sF, ECF_R32_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32_SFLOAT: return convertColor<sF, ECF_R32_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32_UINT: return convertColor<sF, ECF_R32G32_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32_SINT: return convertColor<sF, ECF_R32G32_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32_SFLOAT: return convertColor<sF, ECF_R32G32_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32_UINT: return convertColor<sF, ECF_R32G32B32_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32_SINT: return convertColor<sF, ECF_R32G32B32_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32_SFLOAT: return convertColor<sF, ECF_R32G32B32_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32A32_UINT: return convertColor<sF, ECF_R32G32B32A32_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32A32_SINT: return convertColor<sF, ECF_R32G32B32A32_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32A32_SFLOAT: return convertColor<sF, ECF_R32G32B32A32_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64_UINT: return convertColor<sF, ECF_R64_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64_SINT: return convertColor<sF, ECF_R64_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64_SFLOAT: return convertColor<sF, ECF_R64_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64_UINT: return convertColor<sF, ECF_R64G64_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64_SINT: return convertColor<sF, ECF_R64G64_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64_SFLOAT: return convertColor<sF, ECF_R64G64_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64_UINT: return convertColor<sF, ECF_R64G64B64_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64_SINT: return convertColor<sF, ECF_R64G64B64_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64_SFLOAT: return convertColor<sF, ECF_R64G64B64_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64A64_UINT: return convertColor<sF, ECF_R64G64B64A64_UINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64A64_SINT: return convertColor<sF, ECF_R64G64B64A64_SINT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64A64_SFLOAT: return convertColor<sF, ECF_R64G64B64A64_SFLOAT>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B10G11R11_UFLOAT_PACK32: return convertColor<sF, ECF_B10G11R11_UFLOAT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_E5B9G9R9_UFLOAT_PACK32: return convertColor<sF, ECF_E5B9G9R9_UFLOAT_PACK32>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGB_UNORM_BLOCK: return convertColor<sF, ECF_BC1_RGB_UNORM_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGB_SRGB_BLOCK: return convertColor<sF, ECF_BC1_RGB_SRGB_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGBA_UNORM_BLOCK: return convertColor<sF, ECF_BC1_RGBA_UNORM_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGBA_SRGB_BLOCK: return convertColor<sF, ECF_BC1_RGBA_SRGB_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC2_UNORM_BLOCK: return convertColor<sF, ECF_BC2_UNORM_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC2_SRGB_BLOCK: return convertColor<sF, ECF_BC2_SRGB_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC3_UNORM_BLOCK: return convertColor<sF, ECF_BC3_UNORM_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC3_SRGB_BLOCK: return convertColor<sF, ECF_BC3_SRGB_BLOCK>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8_R8_3PLANE_420_UNORM: return convertColor<sF, ECF_G8_B8_R8_3PLANE_420_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8R8_2PLANE_420_UNORM: return convertColor<sF, ECF_G8_B8R8_2PLANE_420_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8_R8_3PLANE_422_UNORM: return convertColor<sF, ECF_G8_B8_R8_3PLANE_422_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8R8_2PLANE_422_UNORM: return convertColor<sF, ECF_G8_B8R8_2PLANE_422_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8_R8_3PLANE_444_UNORM: return convertColor<sF, ECF_G8_B8_R8_3PLANE_444_UNORM>(_srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        }
    }
    }//namespace impl

    inline void convertColor(ECOLOR_FORMAT _sfmt, ECOLOR_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, uint64_t _scale, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize)
    {
        switch (_sfmt)
        {
        case ECF_A1R5G5B5: return impl::convertColor_RTimpl<ECF_A1R5G5B5>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R5G6B5: return impl::convertColor_RTimpl<ECF_R5G6B5>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R4G4_UNORM_PACK8: return impl::convertColor_RTimpl<ECF_R4G4_UNORM_PACK8>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R4G4B4A4_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_R4G4B4A4_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B4G4R4A4_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_B4G4R4A4_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R5G6B5_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_R5G6B5_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B5G6R5_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_B5G6R5_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R5G5B5A1_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_R5G5B5A1_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B5G5R5A1_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_B5G5R5A1_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A1R5G5B5_UNORM_PACK16: return impl::convertColor_RTimpl<ECF_A1R5G5B5_UNORM_PACK16>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_UNORM: return impl::convertColor_RTimpl<ECF_R8_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SNORM: return impl::convertColor_RTimpl<ECF_R8_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_USCALED: return impl::convertColor_RTimpl<ECF_R8_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SSCALED: return impl::convertColor_RTimpl<ECF_R8_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_UINT: return impl::convertColor_RTimpl<ECF_R8_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SINT: return impl::convertColor_RTimpl<ECF_R8_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8_SRGB: return impl::convertColor_RTimpl<ECF_R8_SRGB>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_UNORM: return impl::convertColor_RTimpl<ECF_R8G8_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SNORM: return impl::convertColor_RTimpl<ECF_R8G8_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_USCALED: return impl::convertColor_RTimpl<ECF_R8G8_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SSCALED: return impl::convertColor_RTimpl<ECF_R8G8_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_UINT: return impl::convertColor_RTimpl<ECF_R8G8_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SINT: return impl::convertColor_RTimpl<ECF_R8G8_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8_SRGB: return impl::convertColor_RTimpl<ECF_R8G8_SRGB>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_UNORM: return impl::convertColor_RTimpl<ECF_R8G8B8_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SNORM: return impl::convertColor_RTimpl<ECF_R8G8B8_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_USCALED: return impl::convertColor_RTimpl<ECF_R8G8B8_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SSCALED: return impl::convertColor_RTimpl<ECF_R8G8B8_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_UINT: return impl::convertColor_RTimpl<ECF_R8G8B8_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SINT: return impl::convertColor_RTimpl<ECF_R8G8B8_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8_SRGB: return impl::convertColor_RTimpl<ECF_R8G8B8_SRGB>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_UNORM: return impl::convertColor_RTimpl<ECF_B8G8R8_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SNORM: return impl::convertColor_RTimpl<ECF_B8G8R8_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_USCALED: return impl::convertColor_RTimpl<ECF_B8G8R8_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SSCALED: return impl::convertColor_RTimpl<ECF_B8G8R8_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_UINT: return impl::convertColor_RTimpl<ECF_B8G8R8_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SINT: return impl::convertColor_RTimpl<ECF_B8G8R8_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8_SRGB: return impl::convertColor_RTimpl<ECF_B8G8R8_SRGB>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_UNORM: return impl::convertColor_RTimpl<ECF_R8G8B8A8_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SNORM: return impl::convertColor_RTimpl<ECF_R8G8B8A8_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_USCALED: return impl::convertColor_RTimpl<ECF_R8G8B8A8_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SSCALED: return impl::convertColor_RTimpl<ECF_R8G8B8A8_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_UINT: return impl::convertColor_RTimpl<ECF_R8G8B8A8_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SINT: return impl::convertColor_RTimpl<ECF_R8G8B8A8_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R8G8B8A8_SRGB: return impl::convertColor_RTimpl<ECF_R8G8B8A8_SRGB>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_UNORM: return impl::convertColor_RTimpl<ECF_B8G8R8A8_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SNORM: return impl::convertColor_RTimpl<ECF_B8G8R8A8_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_USCALED: return impl::convertColor_RTimpl<ECF_B8G8R8A8_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SSCALED: return impl::convertColor_RTimpl<ECF_B8G8R8A8_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_UINT: return impl::convertColor_RTimpl<ECF_B8G8R8A8_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SINT: return impl::convertColor_RTimpl<ECF_B8G8R8A8_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B8G8R8A8_SRGB: return impl::convertColor_RTimpl<ECF_B8G8R8A8_SRGB>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_UNORM_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_UNORM_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SNORM_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_SNORM_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_USCALED_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_USCALED_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SSCALED_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_SSCALED_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_UINT_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_UINT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SINT_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_SINT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A8B8G8R8_SRGB_PACK32: return impl::convertColor_RTimpl<ECF_A8B8G8R8_SRGB_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_UNORM_PACK32: return impl::convertColor_RTimpl<ECF_A2R10G10B10_UNORM_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_SNORM_PACK32: return impl::convertColor_RTimpl<ECF_A2R10G10B10_SNORM_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_USCALED_PACK32: return impl::convertColor_RTimpl<ECF_A2R10G10B10_USCALED_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_SSCALED_PACK32: return impl::convertColor_RTimpl<ECF_A2R10G10B10_SSCALED_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_UINT_PACK32: return impl::convertColor_RTimpl<ECF_A2R10G10B10_UINT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2R10G10B10_SINT_PACK32: return impl::convertColor_RTimpl<ECF_A2R10G10B10_SINT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_UNORM_PACK32: return impl::convertColor_RTimpl<ECF_A2B10G10R10_UNORM_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_SNORM_PACK32: return impl::convertColor_RTimpl<ECF_A2B10G10R10_SNORM_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_USCALED_PACK32: return impl::convertColor_RTimpl<ECF_A2B10G10R10_USCALED_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_SSCALED_PACK32: return impl::convertColor_RTimpl<ECF_A2B10G10R10_SSCALED_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_UINT_PACK32: return impl::convertColor_RTimpl<ECF_A2B10G10R10_UINT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_A2B10G10R10_SINT_PACK32: return impl::convertColor_RTimpl<ECF_A2B10G10R10_SINT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_UNORM: return impl::convertColor_RTimpl<ECF_R16_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SNORM: return impl::convertColor_RTimpl<ECF_R16_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_USCALED: return impl::convertColor_RTimpl<ECF_R16_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SSCALED: return impl::convertColor_RTimpl<ECF_R16_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_UINT: return impl::convertColor_RTimpl<ECF_R16_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SINT: return impl::convertColor_RTimpl<ECF_R16_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16_SFLOAT: return impl::convertColor_RTimpl<ECF_R16_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_UNORM: return impl::convertColor_RTimpl<ECF_R16G16_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SNORM: return impl::convertColor_RTimpl<ECF_R16G16_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_USCALED: return impl::convertColor_RTimpl<ECF_R16G16_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SSCALED: return impl::convertColor_RTimpl<ECF_R16G16_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_UINT: return impl::convertColor_RTimpl<ECF_R16G16_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SINT: return impl::convertColor_RTimpl<ECF_R16G16_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16_SFLOAT: return impl::convertColor_RTimpl<ECF_R16G16_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_UNORM: return impl::convertColor_RTimpl<ECF_R16G16B16_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SNORM: return impl::convertColor_RTimpl<ECF_R16G16B16_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_USCALED: return impl::convertColor_RTimpl<ECF_R16G16B16_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SSCALED: return impl::convertColor_RTimpl<ECF_R16G16B16_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_UINT: return impl::convertColor_RTimpl<ECF_R16G16B16_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SINT: return impl::convertColor_RTimpl<ECF_R16G16B16_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16_SFLOAT: return impl::convertColor_RTimpl<ECF_R16G16B16_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_UNORM: return impl::convertColor_RTimpl<ECF_R16G16B16A16_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SNORM: return impl::convertColor_RTimpl<ECF_R16G16B16A16_SNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_USCALED: return impl::convertColor_RTimpl<ECF_R16G16B16A16_USCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SSCALED: return impl::convertColor_RTimpl<ECF_R16G16B16A16_SSCALED>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_UINT: return impl::convertColor_RTimpl<ECF_R16G16B16A16_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SINT: return impl::convertColor_RTimpl<ECF_R16G16B16A16_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R16G16B16A16_SFLOAT: return impl::convertColor_RTimpl<ECF_R16G16B16A16_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32_UINT: return impl::convertColor_RTimpl<ECF_R32_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32_SINT: return impl::convertColor_RTimpl<ECF_R32_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32_SFLOAT: return impl::convertColor_RTimpl<ECF_R32_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32_UINT: return impl::convertColor_RTimpl<ECF_R32G32_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32_SINT: return impl::convertColor_RTimpl<ECF_R32G32_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32_SFLOAT: return impl::convertColor_RTimpl<ECF_R32G32_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32_UINT: return impl::convertColor_RTimpl<ECF_R32G32B32_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32_SINT: return impl::convertColor_RTimpl<ECF_R32G32B32_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32_SFLOAT: return impl::convertColor_RTimpl<ECF_R32G32B32_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32A32_UINT: return impl::convertColor_RTimpl<ECF_R32G32B32A32_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32A32_SINT: return impl::convertColor_RTimpl<ECF_R32G32B32A32_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R32G32B32A32_SFLOAT: return impl::convertColor_RTimpl<ECF_R32G32B32A32_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64_UINT: return impl::convertColor_RTimpl<ECF_R64_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64_SINT: return impl::convertColor_RTimpl<ECF_R64_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64_SFLOAT: return impl::convertColor_RTimpl<ECF_R64_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64_UINT: return impl::convertColor_RTimpl<ECF_R64G64_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64_SINT: return impl::convertColor_RTimpl<ECF_R64G64_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64_SFLOAT: return impl::convertColor_RTimpl<ECF_R64G64_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64_UINT: return impl::convertColor_RTimpl<ECF_R64G64B64_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64_SINT: return impl::convertColor_RTimpl<ECF_R64G64B64_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64_SFLOAT: return impl::convertColor_RTimpl<ECF_R64G64B64_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64A64_UINT: return impl::convertColor_RTimpl<ECF_R64G64B64A64_UINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64A64_SINT: return impl::convertColor_RTimpl<ECF_R64G64B64A64_SINT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_R64G64B64A64_SFLOAT: return impl::convertColor_RTimpl<ECF_R64G64B64A64_SFLOAT>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_B10G11R11_UFLOAT_PACK32: return impl::convertColor_RTimpl<ECF_B10G11R11_UFLOAT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_E5B9G9R9_UFLOAT_PACK32: return impl::convertColor_RTimpl<ECF_E5B9G9R9_UFLOAT_PACK32>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGB_UNORM_BLOCK: return impl::convertColor_RTimpl<ECF_BC1_RGB_UNORM_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGB_SRGB_BLOCK: return impl::convertColor_RTimpl<ECF_BC1_RGB_SRGB_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGBA_UNORM_BLOCK: return impl::convertColor_RTimpl<ECF_BC1_RGBA_UNORM_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC1_RGBA_SRGB_BLOCK: return impl::convertColor_RTimpl<ECF_BC1_RGBA_SRGB_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC2_UNORM_BLOCK: return impl::convertColor_RTimpl<ECF_BC2_UNORM_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC2_SRGB_BLOCK: return impl::convertColor_RTimpl<ECF_BC2_SRGB_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC3_UNORM_BLOCK: return impl::convertColor_RTimpl<ECF_BC3_UNORM_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_BC3_SRGB_BLOCK: return impl::convertColor_RTimpl<ECF_BC3_SRGB_BLOCK>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8_R8_3PLANE_420_UNORM: return impl::convertColor_RTimpl<ECF_G8_B8_R8_3PLANE_420_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8R8_2PLANE_420_UNORM: return impl::convertColor_RTimpl<ECF_G8_B8R8_2PLANE_420_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8_R8_3PLANE_422_UNORM: return impl::convertColor_RTimpl<ECF_G8_B8_R8_3PLANE_422_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8R8_2PLANE_422_UNORM: return impl::convertColor_RTimpl<ECF_G8_B8R8_2PLANE_422_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        case ECF_G8_B8_R8_3PLANE_444_UNORM: return impl::convertColor_RTimpl<ECF_G8_B8_R8_3PLANE_444_UNORM>(_dfmt, _srcPix, _dstPix, _scale, _pixOrBlockCnt, _imgSize);
        }
    }
	
}} //irr:video

#endif //__IRR_CONVERT_COLOR_H_INCLUDED__