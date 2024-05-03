// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

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

    template<uint32_t alignment>
    __ref<T,alignment,_restrict> deref()
    {
        // TODO: assert(addr&uint64_t(alignment-1)==0);
        using retval_t = __ref < T, alignment, _restrict>;
        retval_t retval;
        retval.__init(impl::bitcast<typename retval_t::spv_ptr_t>(addr));
        return retval;
    }
};
}

#endif