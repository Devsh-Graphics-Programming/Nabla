// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MEMORY_H_INCLUDED__
#define __IRR_MEMORY_H_INCLUDED__

#include "irr/core/math/irrMath.h"
#include "irr/void_t.h"
#include <typeinfo>
#include <cstddef>


#define _IRR_SIMD_ALIGNMENT                 16u // change to 32 or 64 for AVX or AVX2 compatibility respectively, might break BaW file format!
//! Default alignment for a type
#define _IRR_DEFAULT_ALIGNMENT(_obj_type)   (std::alignment_of<_obj_type>::value>(_IRR_SIMD_ALIGNMENT) ? std::alignment_of<_obj_type>::value:(_IRR_SIMD_ALIGNMENT))

#define _IRR_MIN_MAP_BUFFER_ALIGNMENT       64u// GL_MIN_MAP_BUFFER_ALIGNMENT


//! Very useful for enabling compiler optimizations
#if defined(_MSC_VER)
    #define _IRR_ASSUME_ALIGNED(ptr, alignment) \
    __assume((reinterpret_cast<size_t>(ptr) & ((alignment) - 1)) == 0)
#elif (__GNUC__ * 100 + __GNUC_MINOR__) >= 407 // ||(CLANG&&__has_builtin(__builtin_assume_aligned))
    #define _IRR_ASSUME_ALIGNED(ptr, alignment) \
    (ptr) = static_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (alignment)))
#else
    #define _IRR_ASSUME_ALIGNED(ptr,alignment)
#endif

//! Utility so we don't have to write out _IRR_ASSUME_ALIGNED(ptr,_IRR_SIMD_ALIGNMENT) constantly
#define _IRR_ASSUME_SIMD_ALIGNED(ptr) _IRR_ASSUME_ALIGNED(ptr,_IRR_SIMD_ALIGNMENT)


//! You can swap these out for whatever you like, jemalloc, tcmalloc etc. but make them noexcept
#ifdef _IRR_PLATFORM_WINDOWS_
    #define _IRR_ALIGNED_MALLOC(size,alignment)     ::_aligned_malloc(size,alignment)
    #define _IRR_ALIGNED_FREE(addr)                 ::_aligned_free(addr)
#else

namespace irr
{
namespace impl
{
    inline void* aligned_malloc(size_t size, size_t alignment)
    {
        if (size == 0) return nullptr;
        void* p;
        if (::posix_memalign(&p, alignment<alignof(std::max_align_t) ? alignof(std::max_align_t):alignment, size) != 0) p = nullptr;
        return p;
    }
}
}
    #define _IRR_ALIGNED_MALLOC(size,alignment)     irr::impl::aligned_malloc(size,alignment)
    #define _IRR_ALIGNED_FREE(addr)                 ::free(addr)
#endif


namespace irr
{
namespace core
{

//! Alignments can only be PoT in C++11 and in GPU APIs, so this is useful if you need to pad
constexpr inline size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1ull) & ~(alignment - 1ull);
}

//! Down-rounding counterpart
constexpr inline size_t alignDown(size_t value, size_t alignment)
{
    return (value - 1ull) & ~(alignment - 1ull);
}

//! Valid alignments are power of two
constexpr inline bool is_alignment(size_t value)
{
    return core::isPoT(value);
}

//!
constexpr inline bool is_aligned_to(size_t value, size_t alignment)
{
    return core::isPoT(alignment)&&((value&(alignment-1ull))==0ull);
}
constexpr inline bool is_aligned_to(const void* value, size_t alignment)
{
    return core::is_aligned_to(reinterpret_cast<size_t>(value),alignment);
}

}
}


#endif // __IRR_MEMORY_H_INCLUDED__
