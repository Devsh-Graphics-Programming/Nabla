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
    virtual void impl(const double in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
	virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
	virtual void impl(const int64_t in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const { assert(false); } // not override

	virtual void impl(const double in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
	virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
	virtual void impl(const int64_t in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden

	virtual void impl(const double in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
	virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
	virtual void impl(const int64_t in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const { assert(false); } // not overriden
        

	template<typename InT, typename OutT>
	inline void operator()(const InT in[SwizzleBase::MaxChannels], OutT out[SwizzleBase::MaxChannels]) const
	{
		impl(in,out);
	}
};


template<E_FORMAT sF, E_FORMAT dF, class Swizzle = VoidSwizzle>
inline void convertColor(const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY, Swizzle* swizzle = nullptr)
{
    using decT = typename format_interm_storage_type<sF>::type;
    using encT = typename format_interm_storage_type<dF>::type;

    constexpr auto MaxChannels = 4;
    decT decbuf[MaxChannels] = {0, 0, 0, 1};
    encT encbuf[MaxChannels];
    decodePixels<sF>(srcPix,decbuf,_blockX,_blockY);
    if (swizzle)
        swizzle->operator()(decbuf, encbuf);
    encodePixels<dF>(dstPix,encbuf);
}

template<class Swizzle = VoidSwizzle>
inline void convertColor(E_FORMAT sF, E_FORMAT dF, const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY, Swizzle* swizzle = nullptr)
{
    constexpr auto MaxChannels = 4;
    constexpr auto TexelValueSize = MaxChannels*sizeof(uint64_t);
    uint8_t decbuf[TexelValueSize];
    uint8_t encbuf[TexelValueSize];

    decodePixelsRuntime(sF, srcPix, decbuf, _blockX, _blockY);
    if (swizzle)
        swizzle->operator()(decbuf, encbuf);
    encodePixelsRuntime(dF, dstPix, encbuf);
}


}
}

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#endif //__IRR_CONVERT_COLOR_H_INCLUDED__
