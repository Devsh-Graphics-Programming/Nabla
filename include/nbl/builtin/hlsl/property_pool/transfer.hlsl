
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#ifndef _NBL_HLSL_PROPERTY_POOL_TRANSFER_HLSL_INCLUDED_
#define _NBL_HLSL_PROPERTY_POOL_TRANSFER_HLSL_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace property_pool
{


struct transfer_t
{
    int propertyDWORDsize_flags;
    int elementCount;
    uint srcIndexOffset;
    uint dstIndexOffset;
};

#define TRANSFER_EF_DOWNLOAD 0x1u
#define TRANSFER_EF_SRC_FILL 0x2u
#define TRANSFER_EF_BIT_COUNT 2

#define INVALID 0xdeadbeef

#define MAX_PROPERTIES_PER_DISPATCH 128


}
}
}

#endif