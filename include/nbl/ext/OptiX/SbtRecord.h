// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_EXT_SBT_RECORD_H_INCLUDED__
#define __NBL_EXT_SBT_RECORD_H_INCLUDED__

#include "optix.h"

namespace nbl
{
namespace ext
{
namespace OptiX
{
template<typename T>
struct SbtRecord
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

}
}
}

#endif