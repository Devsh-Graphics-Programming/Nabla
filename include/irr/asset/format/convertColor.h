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

namespace irr
{
namespace asset
{
    namespace impl
    {
        struct E_TYPE
        {
            struct ET_I64
            {
                typedef int64_t type;
            };
            struct ET_U64
            {
                typedef uint64_t type;
            };
            struct ET_F64
            {
                typedef double type;
            };
        };

        template<asset::E_FORMAT cf>
        struct format2type :
            std::conditional<asset::isIntegerFormat<cf>(), 
                typename std::conditional<asset::isSignedFormat<cf>(),E_TYPE::ET_I64,E_TYPE::ET_U64>::type,
                typename std::conditional<asset::isFloatingPointFormat<cf>()||asset::isNormalizedFormat<cf>()||asset::isScaledFormat<cf>(),E_TYPE::ET_F64,void>::type // gen error if format is neither signed/unsigned integer or floating point/normalized/scaled
            >::type
        {
        };
    } //namespace impl
    

    template<asset::E_FORMAT cf, class fmt_class=impl::format2type<cf>>
    struct SCallDecode
    {
        inline void operator()(const void* _pix[4], typename fmt_class::type* _output, uint32_t _blockX, uint32_t _blockY)
        {
            static_assert(!std::is_void<fmt_class::type>::value, "Logic Error in Metaprogramming code!");
            decodePixels<cf,typename fmt_class::type>(_pix, _output, _blockX, _blockY);
            }
    };

    template<asset::E_FORMAT cf, class fmt_class=impl::format2type<cf>::value>
    struct SCallEncode
    {
        inline void operator()(void* _pix, const typename fmt_class::type* _input)
        {
            static_assert(!std::is_void<fmt_class::type>::value, "Logic Error in Metaprogramming code!");
            encodePixels<cf, typename fmt_class::type>(_pix, _input);
        }
    };


    struct SwizzleBase
    {
        _IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;
    };

	struct VoidSwizzle : SwizzleBase
	{
		template<typename InT, typename OutT>
		inline void operator()(const InT in[SwizzleBase::MaxChannels], OutT out[SwizzleBase::MaxChannels]) const
        {
            std::fill(out,out+4,in);
        }
	};

	struct PolymorphicSwizzle : SwizzleBase
	{
		virtual void impl(const double in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const = 0;
		virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const = 0;
		virtual void impl(const int64_t in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const = 0;

		virtual void impl(const double in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const = 0;
		virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const = 0;
		virtual void impl(const int64_t in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const = 0;

		virtual void impl(const double in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const = 0;
		virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const = 0;
		virtual void impl(const int64_t in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const = 0;
        

		template<typename InT, typename OutT>
		inline void operator()(const InT in[SwizzleBase::MaxChannels], OutT out[SwizzleBase::MaxChannels]) const
		{
			impl(in,out);
		}
	};


    template<asset::E_FORMAT sF, asset::E_FORMAT dF, class Swizzle = VoidSwizzle >
    inline void convertColor(const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY, PolymorphicSwizzle* swizzle = nullptr)
    {
        using decT = typename impl::format2type<sF>::type;
        using encT = typename impl::format2type<dF>::type;

        constexpr auto MaxChannels = 4;
        decT decbuf[MaxChannels] = {0, 0, 0, 1};
        encT encbuf[MaxChannels];
        impl::SCallDecode<sF>{}(srcPix, decbuf, _blockX, _blockY);
        if (!std::is_void<Swizzle>::value)
            state->operator()(decbuf, encbuf);
        else if (swizzle)
            swizzle->operator()(decbuf, encbuf);
        impl::SCallEncode<dF>{}(dstPix, encbuf);
    }

}
}

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#endif //__IRR_CONVERT_COLOR_H_INCLUDED__
