// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_RAW_BUFFER_BLOB_H_INCLUDED__
#define __IRR_RAW_BUFFER_BLOB_H_INCLUDED__

#include "irr/asset/bawformat/Blob.h"

namespace irr
{
namespace asset
{

class ICPUBuffer;

#include "irr/irrpack.h"
struct IRR_FORCE_EBO RawBufferBlobV0 : TypedBlob<RawBufferBlobV0, ICPUBuffer>, VariableSizeBlob<RawBufferBlobV0, ICPUBuffer>
{};
#include "irr/irrunpack.h"

using RawBufferBlobV1 = RawBufferBlobV0;
using RawBufferBlobV2 = RawBufferBlobV1;
using RawBufferBlobV3 = RawBufferBlobV2;

template<>
struct CorrespondingBlobTypeFor<ICPUBuffer> { typedef RawBufferBlobV3 type; };

}
} // irr::asset

#endif
