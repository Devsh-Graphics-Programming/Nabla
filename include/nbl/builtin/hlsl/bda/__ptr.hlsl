// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/bda/__ref.hlsl"

#ifndef _NBL_BUILTIN_HLSL_BDA_PTR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_PTR_INCLUDED_

namespace bda
{
template<typename T, bool _restrict>
struct __ptr
{
    using this_t = __ptr < T, _restrict>;
    uint64_t addr;

    static this_t create(const uint64_t _addr)
    {
        this_t retval;
        retval.addr = _addr;
        return retval;
    }

    template<uint64_t alignment=nbl::hlsl::alignment_of<T>::value>
    __ref<T,alignment,_restrict> deref()
    {
        // TODO: assert(addr&uint64_t(alignment-1)==0);
        using retval_t = __ref < T, alignment, _restrict>;
        retval_t retval;
        retval.__init(impl::bitcast<typename retval_t::spv_ptr_t>(addr));
        return retval;
    }
};

template<typename T>
struct PtrAccessor
{
    static PtrAccessor createAccessor(uint64_t addr)
    {
        PtrAccessor ptr;
        ptr.addr = addr;
        return ptr;
    }

    T get(uint64_t index)
    {
        return __ptr<T>(addr + sizeof(T) * index).template deref().load();
    }

    void set(uint64_t index, T value)
    {
        __ptr<T>(addr + sizeof(T) * index).template deref().store(value);
    }

    T atomicAdd(uint64_t index, T value)
    {
        return __ptr<T>(addr + sizeof(uint32_t) * index).template deref().atomicAdd(value);
    }

    uint64_t addr;
};

}

#endif