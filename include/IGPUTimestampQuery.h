// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_TIMESTAMP_QUERY_H_INCLUDED__
#define __NBL_I_GPU_TIMESTAMP_QUERY_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "stdint.h"

namespace nbl
{
namespace video
{
class IGPUTimestampQuery : public core::IReferenceCounted
{
    _NBL_INTERFACE_CHILD(IGPUTimestampQuery) {}

public:
    virtual bool isQueryReady() = 0;

    virtual uint64_t getTimestampWhenCompleted() = 0;
};

}
}

#endif
