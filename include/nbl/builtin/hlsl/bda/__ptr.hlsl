// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/bda/__ref.hlsl"

#ifndef _NBL_BUILTIN_HLSL_BDA_PTR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_PTR_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace bda
{
template<typename T>
struct __ptr
{
    using this_t = __ptr<T>;
    uint32_t2 addr;

    static this_t create(const uint32_t2 _addr)
    {
        this_t retval;
        retval.addr = _addr;
        return retval;
    }

    // in non-64bit mode we only support "small" arithmetic on pointers (just offsets no arithmetic on pointers)
    __ptr operator+(uint32_t i)
    {
        i *= sizeof(T);
        uint32_t2 newAddr = addr;
        AddCarryOutput<T> lsbAddRes = spirv::addCarry<uint32_t>(addr[0],i);
        newAddr[0] = lsbAddRes.result;
        newAddr[1] += lsbAddRes.carry;
        return __ptr::create(newAddr);
    }
    __ptr operator-(uint32_t i)
    {
        i *= sizeof(T);
        uint32_t2 newAddr = addr;
        AddCarryOutput<T> lsbSubRes = spirv::subBorrow<uint32_t>(addr[0],i);
        newAddr[0] = lsbSubRes.result;
        newAddr[1] -= lsbSubRes.carry;
        return __ptr::create(newAddr);
    }

    template< uint64_t alignment=alignment_of_v<T> >
    __ref<T,alignment,false> deref()
    {
        // TODO: assert(addr&uint64_t(alignment-1)==0);
        __ref<T,alignment,false> retval;
        retval.__init(spirv::bitcast<spirv::bda_pointer_t<T>,uint32_t2>(addr));
        return retval;
    }

    //! Dont use these, to avoid emitting shaderUint64 capability when compiling for crappy mobile GPUs
    static this_t create(const uint64_t _addr)
    {
        this_t retval;
        retval.addr = spirv::bitcast<uint32_t2>(_addr);
        return retval;
    }
    __ptr operator+(int64_t i)
    {
        i *= sizeof(T);
        return __ptr::create(spirv::bitcast<uint64_t>(addr)+i);
    }
    __ptr operator-(int64_t i)
    {
        i *= sizeof(T);
        return __ptr::create(spirv::bitcast<uint64_t>(addr)-i);
    }
};

}
}
}

#endif