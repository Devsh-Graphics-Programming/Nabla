// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_MEMORY_H_INCLUDED__
#define __NBL_CORE_MEMORY_H_INCLUDED__

#include "nbl/core/math/intutil.h"

#include <typeinfo>
#include <cstddef>


#define _NBL_SIMD_ALIGNMENT                 16u // change to 32 or 64 for AVX or AVX2 compatibility respectively, might break BaW file format!
//! Default alignment for a type
#define _NBL_DEFAULT_ALIGNMENT(_obj_type)   (std::alignment_of<_obj_type>::value>(_NBL_SIMD_ALIGNMENT) ? std::alignment_of<_obj_type>::value:(_NBL_SIMD_ALIGNMENT))

#define _NBL_MIN_MAP_BUFFER_ALIGNMENT       64u// GL_MIN_MAP_BUFFER_ALIGNMENT


//! Very useful for enabling compiler optimizations
#if defined(_MSC_VER)
    #define _NBL_ASSUME_ALIGNED(ptr, alignment) \
    __assume((reinterpret_cast<size_t>(ptr) & ((alignment) - 1)) == 0)
#elif (__GNUC__ * 100 + __GNUC_MINOR__) >= 407 // ||(CLANG&&__has_builtin(__builtin_assume_aligned))
    #define _NBL_ASSUME_ALIGNED(ptr, alignment) \
    (ptr) = static_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (alignment)))
#else
    #define _NBL_ASSUME_ALIGNED(ptr,alignment)
#endif

//! Utility so we don't have to write out _NBL_ASSUME_ALIGNED(ptr,_NBL_SIMD_ALIGNMENT) constantly
#define _NBL_ASSUME_SIMD_ALIGNED(ptr) _NBL_ASSUME_ALIGNED(ptr,_NBL_SIMD_ALIGNMENT)


//! You can swap these out for whatever you like, jemalloc, tcmalloc etc. but make them noexcept
#ifdef _NBL_PLATFORM_WINDOWS_
    #define _NBL_ALIGNED_MALLOC(size,alignment)     ::_aligned_malloc(size,alignment)
    #define _NBL_ALIGNED_FREE(addr)                 ::_aligned_free(addr)
#else

namespace nbl
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
    #define _NBL_ALIGNED_MALLOC(size,alignment)     nbl::impl::aligned_malloc(size,alignment)
    #define _NBL_ALIGNED_FREE(addr)                 ::free(addr)
#endif


namespace nbl
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
// clang complains about constexpr so make normal for now (also complains abour reinterpret_cast)
inline bool is_aligned_to(const void* value, size_t alignment)
{
    return core::is_aligned_to(reinterpret_cast<size_t>(value),alignment);
}

}
}


#endif
