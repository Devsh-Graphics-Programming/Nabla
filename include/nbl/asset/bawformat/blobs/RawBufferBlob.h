// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_RAW_BUFFER_BLOB_H_INCLUDED__
#define __NBL_ASSET_RAW_BUFFER_BLOB_H_INCLUDED__

#include "nbl/asset/bawformat/Blob.h"

namespace nbl
{
namespace asset
{
class ICPUBuffer;

#include "nbl/nblpack.h"
struct NBL_FORCE_EBO RawBufferBlobV0 : TypedBlob<RawBufferBlobV0, ICPUBuffer>, VariableSizeBlob<RawBufferBlobV0, ICPUBuffer>
{
};
#include "nbl/nblunpack.h"

using RawBufferBlobV1 = RawBufferBlobV0;
using RawBufferBlobV2 = RawBufferBlobV1;
using RawBufferBlobV3 = RawBufferBlobV2;

template<>
struct CorrespondingBlobTypeFor<ICPUBuffer>
{
    typedef RawBufferBlobV3 type;
};

}
}  // nbl::asset

#endif
