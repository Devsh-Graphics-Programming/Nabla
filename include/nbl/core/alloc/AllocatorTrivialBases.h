// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ALLOCATOR_TRIVIAL_BASES_H_INCLUDED__
#define __NBL_CORE_ALLOCATOR_TRIVIAL_BASES_H_INCLUDED__

namespace nbl
{
namespace core
{
template<typename T>
class AllocatorTrivialBase;

template<>
class NBL_FORCE_EBO AllocatorTrivialBase<void>
{
public:
    typedef void value_type;
    typedef void* pointer;
    typedef const void* const_pointer;

    typedef void* void_pointer;
    typedef const void* const_void_pointer;
};

template<typename T>
class NBL_FORCE_EBO AllocatorTrivialBase
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

    typedef void* void_pointer;
    typedef const void* const_void_pointer;
};

}
}

#endif
