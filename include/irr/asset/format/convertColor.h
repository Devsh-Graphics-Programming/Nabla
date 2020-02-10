#ifndef __IRR_CONVERT_COLOR_H_INCLUDED__
#define __IRR_CONVERT_COLOR_H_INCLUDED__

#include <cassert>
#include <type_traits>

#include "irr/static_if.h"
#include "irr/asset/format/EFormat.h"
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


	struct DefaultSwizzle
	{
		template<typename type>
		constexpr void operator()(type vect[4]) const {}
	};
	struct PolymorphicSwizzle
	{
		virtual void impl(double vect[4]) const = 0;
		virtual void impl(uint64_t vect[4]) const = 0;
		virtual void impl(int64_t vect[4]) const = 0;

		template<typename type>
		inline void operator()(type vect[4]) const
		{
			impl(vect);
		}
	};

    template<asset::E_FORMAT sF, asset::E_FORMAT dF, class Swizzle = DefaultSwizzle >
    inline void convertColor(const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY, PolymorphicSwizzle* swizzle = nullptr)
    {
	#define SWIZZLE(X) \
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_void<Swizzle>::value) \
		{ \
			if (swizzle) \
				swizzle->operator()(X); \
		} \
		IRR_PSEUDO_ELSE_CONSTEXPR \
		{ \
			Swizzle().operator()(X); \
		} \
		IRR_PSEUDO_IF_CONSTEXPR_END 

        using namespace asset;
        if (isIntegerFormat<sF>() && isIntegerFormat<dF>())
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4] {0, 0, 0, 1};
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
			SWIZZLE(decbuf)
            impl::SCallEncode<dF, encT>{}(dstPix, reinterpret_cast<encT*>(decbuf));
        }
        else if (
            (isNormalizedFormat<sF>() || isScaledFormat<sF>() || isFloatingPointFormat<sF>()) && (isNormalizedFormat<dF>() || isScaledFormat<dF>() || isFloatingPointFormat<dF>())
        )
        {
            using decT = double;
            using encT = double;

            decT decbuf[4] { 0, 0, 0, 1 };
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
			SWIZZLE(decbuf)
            impl::SCallEncode<dF, encT>{}(dstPix, decbuf);
        }
        else if ((isFloatingPointFormat<sF>() || isScaledFormat<sF>() || isNormalizedFormat<sF>()) && isIntegerFormat<dF>())
        {
            using decT = double;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4] { 0, 0, 0, 1 };
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
			SWIZZLE(decbuf)
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = static_cast<encT>(decbuf[i]);
            impl::SCallEncode<dF, encT>{}(dstPix, encbuf);
        }
        else if (isIntegerFormat<sF>() && (isNormalizedFormat<dF>() || isScaledFormat<dF>() || isFloatingPointFormat<dF>()))
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = double;

            decT decbuf[4] { 0, 0, 0, 1 };
            impl::SCallDecode<sF, decT>{}(srcPix, decbuf, _blockX, _blockY);
			SWIZZLE(decbuf)
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            impl::SCallEncode<dF, encT>{}(dstPix, encbuf);
        }
	#undef SWIZZLE
    }
	//! A function converting a data to desired texel format
	/**
		To use it, you have to pass source data with texel format \bsF\b you want to exchange
		with \bdF\b. srcPix data is an array due to planar formats. Normally you would pass
		to it data dived into 4 pointers with single channel data per pointer to each array element, 
		but if source data isn't a planar format, you have to pass \awhole\a data to \bsrcPix[0]\b without 
		caring about left elements - make them nullptr. \bdstPix\b is a destination pointer that you will
        use after convertion. Remember - you have to carry about it's size before passing it to the 
        function, so if were to make it RGBA beginning with R, you would have to allocate appropriate memory for RGBA buffer.
		\b_pixOrBlockCnt\b is an amount of texels or blocks you want to convert and \b_imgSize\b is a size
		in texels of an image they belong to. There is also a polymorphic \bswizzle\b parameter
		that makes the whole process slower if used (otherwise it is a null pointer), but it 
		allows you to swizzle the RGBA compoments into a different arrangement at runtime.

		So for example, if a texel amount is 4 and a data arrangement passed to the function looks like
		\aR, R, R, R\a, you could convert it for instance to a data arrangement looking like
		\bRGBA, RGBA, RGBA, RGBA\a. As you may see texel amount doesn't change, but the internal
		buffer's layout does as desired.
	*/
    template<asset::E_FORMAT sF, asset::E_FORMAT dF, class Swizzle = DefaultSwizzle >
    inline void convertColor(const void* srcPix[4], void* dstPix, size_t _pixOrBlockCnt, const core::vector3d<uint32_t>& _imgSize, PolymorphicSwizzle* swizzle = nullptr)
    {
        using namespace asset;

        const uint32_t srcStride = getTexelOrBlockBytesize(sF);
        const uint32_t dstStride = getTexelOrBlockBytesize(dF);

        uint32_t hPlaneReduction[4], vPlaneReduction[4], chCntInPlane[4];
        getHorizontalReductionFactorPerPlane(sF, hPlaneReduction);
        getVerticalReductionFactorPerPlane(sF, vPlaneReduction);
        getChannelsPerPlane(sF, chCntInPlane);

        const auto sdims = getBlockDimensions(sF);

        const uint8_t** src = reinterpret_cast<const uint8_t**>(srcPix);
        uint8_t* const dst_begin = reinterpret_cast<uint8_t*>(dstPix);
        for (size_t i = 0u; i < _pixOrBlockCnt; ++i)
        {
            // assuming _imgSize is always represented in texels
            const auto px = static_cast<uint32_t>(i % size_t(_imgSize.X / sdims.X));
            const auto py = static_cast<uint32_t>(i / size_t(_imgSize.X / sdims.X));
            //px, py are block or texel position
            //x, y are position within block
            for (uint32_t x = 0u; x < sdims.X; ++x)
            {
                for (uint32_t y = 0u; y < sdims.Y; ++y)
                {
                    const ptrdiff_t off = ((sdims.Y * py + y)*_imgSize.X + px * sdims.X + x);
                    convertColor<sF, dF, Swizzle>(reinterpret_cast<const void**>(src), dst_begin + static_cast<ptrdiff_t>(dstStride)*off, x, y, swizzle);
                }
            }
            if (!isPlanarFormat<sF>())
            {
                src[0] += srcStride;
            }
            else
            {
                const uint32_t px = static_cast<uint32_t>(i % size_t(_imgSize.X));
                const uint32_t py = static_cast<uint32_t>(i / size_t(_imgSize.X));
                for (uint32_t j = 0u; j < 4u; ++j)
                    src[j] = reinterpret_cast<const uint8_t*>(srcPix[j]) + chCntInPlane[j]*((_imgSize.X/hPlaneReduction[j]) * (py/vPlaneReduction[j]) + px/hPlaneReduction[j]);
            }
        }
    }


    void convertColor(asset::E_FORMAT _sfmt, asset::E_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, size_t _pixOrBlockCnt, core::vector3d<uint32_t>& _imgSize, PolymorphicSwizzle* swizzle=nullptr);
}} //irr:video

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#endif //__IRR_CONVERT_COLOR_H_INCLUDED__