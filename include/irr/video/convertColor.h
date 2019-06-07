#ifndef __IRR_CONVERT_COLOR_H_INCLUDED__
#define __IRR_CONVERT_COLOR_H_INCLUDED__

#include <cassert>
#include "irr/static_if.h"
#include "irr/asset/EFormat.h"
#include "decodePixels.h"
#include "encodePixels.h"

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wuninitialized"
#endif

namespace irr { namespace video
{
    namespace impl
    {
        enum E_TYPE
        {
            ET_I64,
            ET_U64,
            ET_F64
        };

        template<asset::E_FORMAT cf>
        struct format2type :
            std::conditional<asset::isIntegerFormat<cf>(), 
                typename std::conditional<asset::isSignedFormat<cf>(),
                    std::integral_constant<E_TYPE, ET_I64>,
                    std::integral_constant<E_TYPE, ET_U64>
                >::type,
                typename std::conditional<asset::isFloatingPointFormat<cf>() || asset::isNormalizedFormat<cf>() || asset::isScaledFormat<cf>(),
                    std::integral_constant<E_TYPE, ET_F64>,
                    void // gen error if format is neither signed/unsigned integer or floating point/normalized/scaled
                >::type
            >::type
        {};

        template<asset::E_FORMAT cf, typename T, E_TYPE fmt_class = format2type<cf>::value>
        struct SCallDecode
        {
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY)
            {
                constexpr bool valid =  (std::is_floating_point_v<T>&&fmt_class==ET_F64)||
                                        (std::is_signed_v<T>&&fmt_class==ET_I64)||
                                        (std::is_unsigned_v<T>&&fmt_class==ET_U64);
                IRR_PSEUDO_IF_CONSTEXPR_BEGIN(valid)
                    decodePixels<cf, T>(_pix, _output, _blockX, _blockY);
                IRR_PSEUDO_ELSE_CONSTEXPR 
                    assert(0);
                IRR_PSEUDO_IF_CONSTEXPR_END
            }
        };



        template<asset::E_FORMAT cf, typename T, E_TYPE fmt_class = format2type<cf>::value>
        struct SCallEncode
        {
            inline void operator()(void* _pix, const T* _input)
            {
                constexpr bool valid =  (std::is_floating_point_v<T>&&fmt_class==ET_F64)||
                                        (std::is_signed_v<T>&&fmt_class==ET_I64)||
                                        (std::is_unsigned_v<T>&&fmt_class==ET_U64);
                IRR_PSEUDO_IF_CONSTEXPR_BEGIN(valid)
                    encodePixels<cf, T>(_pix, _input);
                IRR_PSEUDO_ELSE_CONSTEXPR 
                    assert(0);
                IRR_PSEUDO_IF_CONSTEXPR_END
            }
        };
    } //namespace impl


    template<asset::E_FORMAT sF, asset::E_FORMAT dF>
    inline void convertColor(const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY)
    {
        using namespace asset;
        if (isIntegerFormat<sF>() && isIntegerFormat<dF>())
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
            impl::SCallEncode<dF, encT>{}(dstPix, reinterpret_cast<encT*>(decbuf));
        }
        else if (
            (isNormalizedFormat<sF>() || isScaledFormat<sF>() || isFloatingPointFormat<sF>()) && (isNormalizedFormat<dF>() || isScaledFormat<dF>() || isFloatingPointFormat<dF>())
        )
        {
            using decT = double;
            using encT = double;

            decT decbuf[4];
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
            impl::SCallEncode<dF, encT>{}(dstPix, decbuf);
        }
        else if ((isFloatingPointFormat<sF>() || isScaledFormat<sF>() || isNormalizedFormat<sF>()) && isIntegerFormat<dF>())
        {
            using decT = double;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            impl::SCallEncode<dF, encT>{}(dstPix, encbuf);
        }
        else if (isIntegerFormat<sF>() && (isNormalizedFormat<dF>() || isScaledFormat<dF>() || isFloatingPointFormat<dF>()))
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = double;

            decT decbuf[4];
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            impl::SCallEncode<dF, encT>{}(dstPix, encbuf);
        }
    }
    template<asset::E_FORMAT sF, asset::E_FORMAT dF>
    inline void convertColor(const void* srcPix[4], void* dstPix, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize)
    {
        using namespace asset;

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
                {
                    const ptrdiff_t off = ((sdims.Y * py + y)*_imgSize.X + px * sdims.X + x);
                    convertColor<sF, dF>(reinterpret_cast<const void**>(src), dst_begin + static_cast<ptrdiff_t>(dstStride)*off, x, y);
                }
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

    void convertColor(asset::E_FORMAT _sfmt, asset::E_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize);
}} //irr:video

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#endif //__IRR_CONVERT_COLOR_H_INCLUDED__