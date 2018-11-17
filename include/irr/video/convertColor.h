#ifndef __IRR_CONVERT_COLOR_H_INCLUDED__
#define __IRR_CONVERT_COLOR_H_INCLUDED__

#include <cassert>
#include "EColorFormat.h"
#include "decodePixels.h"
#include "encodePixels.h"

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

        template<ECOLOR_FORMAT cf>
        struct format2type :
            std::conditional<isIntegerFormat<cf>(), 
                typename std::conditional<isSignedFormat<cf>(),
                    std::integral_constant<E_TYPE, ET_I64>,
                    std::integral_constant<E_TYPE, ET_U64>
                >::type,
                typename std::conditional<isFloatingPointFormat<cf>() || isNormalizedFormat<cf>() || isScaledFormat<cf>(),
                    std::integral_constant<E_TYPE, ET_F64>,
                    void // gen error if format is neither signed/unsigned integer or floating point/normalized/scaled
                >::type
            >::type
        {};

        template<bool SCALED, ECOLOR_FORMAT cf, typename T, E_TYPE = format2type<cf>::value>
        struct SCallDecode;

        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, double, ET_F64>
        {
            using T = double;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, double, ET_I64>
        {
            using T = double;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, double, ET_U64>
        {
            using T = double;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, int64_t, ET_F64>
        {
            using T = int64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, int64_t, ET_I64>
        {
            using T = int64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, int64_t, ET_U64>
        {
            using T = int64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, uint64_t, ET_F64>
        {
            using T = uint64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, uint64_t, ET_I64>
        {
            using T = uint64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<false, cf, uint64_t, ET_U64>
        {
            using T = uint64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY);
            }
        };


        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, double, ET_F64>
        {
            using T = double;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY, _scale);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, double, ET_I64>
        {
            using T = double;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, double, ET_U64>
        {
            using T = double;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, int64_t, ET_F64>
        {
            using T = int64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, int64_t, ET_I64>
        {
            using T = int64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY, _scale);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, int64_t, ET_U64>
        {
            using T = int64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, uint64_t, ET_F64>
        {
            using T = uint64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, uint64_t, ET_I64>
        {
            using T = uint64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallDecode<true, cf, uint64_t, ET_U64>
        {
            using T = uint64_t;
            inline void operator()(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
            {
                decodePixels<cf, T>(_pix, _output, _blockX, _blockY, _scale);
            }
        };



        template<bool SCALED, ECOLOR_FORMAT cf, typename T, E_TYPE = format2type<cf>::value>
        struct SCallEncode;

        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, double, ET_F64>
        {
            using T = double;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                encodePixels<cf, T>(_pix, _input);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, double, ET_I64>
        {
            using T = double;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, double, ET_U64>
        {
            using T = double;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, int64_t, ET_F64>
        {
            using T = int64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, int64_t, ET_I64>
        {
            using T = int64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                encodePixels<cf, T>(_pix, _input);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, int64_t, ET_U64>
        {
            using T = int64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, uint64_t, ET_F64>
        {
            using T = uint64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, uint64_t, ET_I64>
        {
            using T = uint64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<false, cf, uint64_t, ET_U64>
        {
            using T = uint64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t)
            {
                encodePixels<cf, T>(_pix, _input);
            }
        };

        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, double, ET_F64>
        {
            using T = double;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                encodePixels<cf, T>(_pix, _input, _scale);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, double, ET_I64>
        {
            using T = double;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, double, ET_U64>
        {
            using T = double;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, int64_t, ET_F64>
        {
            using T = int64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, int64_t, ET_I64>
        {
            using T = int64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                encodePixels<cf, T>(_pix, _input, _scale);
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, int64_t, ET_U64>
        {
            using T = int64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, uint64_t, ET_F64>
        {
            using T = uint64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, uint64_t, ET_I64>
        {
            using T = uint64_t;
            inline void operator()(void* _pix, const T* _input, uint64_t _scale)
            {
                assert(0);
                //this never gets called but must be defined because of runtime if-statements (we don't support c++17 and so can't use constexpr-if)
            }
        };
        template<ECOLOR_FORMAT cf>
        struct SCallEncode<true, cf, uint64_t, ET_U64>
        {
            using T = uint64_t;
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
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
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
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, decbuf, _scale);
        }
        else if ((isFloatingPointFormat<sF>() || isNormalizedFormat<sF>()) && isIntegerFormat<dF>())
        {
            using decT = double;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, encbuf, _scale);
        }
        else if (isIntegerFormat<sF>() && (isNormalizedFormat<dF>() || isFloatingPointFormat<dF>()))
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = double;

            decT decbuf[4];
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _blockX, _blockY, _scale);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
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
                {
                    const ptrdiff_t off = ((sdims.Y * py + y)*_imgSize.X + px * sdims.X + x);
                    convertColor<sF, dF>(reinterpret_cast<const void**>(src), dst_begin + static_cast<ptrdiff_t>(dstStride)*off, _scale, x, y);
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

    void convertColor(ECOLOR_FORMAT _sfmt, ECOLOR_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, uint64_t _scale, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize);
}} //irr:video

#endif //__IRR_CONVERT_COLOR_H_INCLUDED__