// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/bda/__ref.hlsl"

#ifndef _NBL_BUILTIN_HLSL_MEMORY_INCLUDED_
#define _NBL_BUILTIN_HLSL_MEMORY_INCLUDED_

namespace nbl
{
namespace hlsl
{

template<typename T, uint32_t Alignment, bool _restrict>
bda::__ptr<T> pointer_to(bda::__ref<T,Alignment,_restrict>) {
    bda::__ptr<T> retval;
    retval.addr = spirv::bitcast<uint64_t>(ptr);
    return retval;    
}
       
}
}

#endif