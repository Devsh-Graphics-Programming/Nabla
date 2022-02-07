// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_CONVERT_COLOR_H_INCLUDED__
#define __NBL_ASSET_CONVERT_COLOR_H_INCLUDED__

#include <cassert>
#include <type_traits>

#include "nbl/asset/format/EFormat.h"
#include "decodePixels.h"
#include "encodePixels.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

namespace nbl
{
namespace asset
{
struct SwizzleBase
{
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;
};

struct VoidSwizzle : SwizzleBase
{
    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const;
};

template<>
inline void VoidSwizzle::operator()<void, void>(const void* in, void* out) const
{
    memcpy(out, in, sizeof(uint64_t) * SwizzleBase::MaxChannels);
}

template<typename InT, typename OutT>
inline void VoidSwizzle::operator()(const InT* in, OutT* out) const
{
    std::copy<const InT*, OutT*>(in, in + 4, out);
}

/*
    Base class for \bruntime\b swizzle - stateful
*/

struct PolymorphicSwizzle : SwizzleBase
{
    virtual void impl(const double in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden
    virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden
    virtual void impl(const int64_t in[SwizzleBase::MaxChannels], double out[SwizzleBase::MaxChannels]) const { assert(false); }  // not override

    virtual void impl(const double in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden
    virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden
    virtual void impl(const int64_t in[SwizzleBase::MaxChannels], uint64_t out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden

    virtual void impl(const double in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden
    virtual void impl(const uint64_t in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden
    virtual void impl(const int64_t in[SwizzleBase::MaxChannels], int64_t out[SwizzleBase::MaxChannels]) const { assert(false); }  // not overriden

    virtual void impl(const void* in, void* out) const { assert(false); }  // not overriden

    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out) const;
};

template<>
inline void PolymorphicSwizzle::operator()<void, void>(const void* in, void* out) const
{
    impl(in, out);
}

template<typename InT, typename OutT>
inline void PolymorphicSwizzle::operator()(const InT* in, OutT* out) const
{
    impl(in, out);
}

template<E_FORMAT sF, E_FORMAT dF, class Swizzle = VoidSwizzle>
inline void convertColor(const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY, const Swizzle& swizzle = Swizzle())
{
    using decT = typename format_interm_storage_type<sF>::type;
    using encT = typename format_interm_storage_type<dF>::type;

    constexpr auto MaxChannels = 4;
    decT decbuf[MaxChannels] = {0, 0, 0, 1};
    encT encbuf[MaxChannels];
    decodePixels<sF>(srcPix, decbuf, _blockX, _blockY);
    swizzle.template operator()<decT, encT>(decbuf, encbuf);
    encodePixels<dF>(dstPix, encbuf);
}

template<class Swizzle = VoidSwizzle>
inline void convertColor(E_FORMAT sF, E_FORMAT dF, const void* srcPix[4], void* dstPix, uint32_t _blockX, uint32_t _blockY, const Swizzle& swizzle = Swizzle())
{
    constexpr auto MaxChannels = 4;
    constexpr auto TexelValueSize = MaxChannels * sizeof(uint64_t);
    uint8_t decbuf[TexelValueSize];
    uint8_t encbuf[TexelValueSize];

    decodePixelsRuntime(sF, srcPix, decbuf, _blockX, _blockY);
    swizzle.template operator()<void, void>(decbuf, encbuf);
    encodePixelsRuntime(dF, dstPix, encbuf);
}

}
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif
