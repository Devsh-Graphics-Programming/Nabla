// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_MESH_BLOB_H_INCLUDED__
#define __IRR_MESH_BLOB_H_INCLUDED__

#include "irr/asset/bawformat/Blob.h"

namespace irr
{
namespace asset
{

class ICPUMesh;

#include "irr/irrpack.h"
//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
struct IRR_FORCE_EBO MeshBlobV0 : VariableSizeBlob<MeshBlobV0,asset::ICPUMesh>, TypedBlob<MeshBlobV0, asset::ICPUMesh>
{
	//friend struct SizedBlob<VariableSizeBlob, MeshBlobV0, asset::ICPUMesh>;
public:
        //! WARNING: Constructor saves only bounding box and mesh buffer count (not mesh buffer pointers)
	explicit MeshBlobV0(const asset::ICPUMesh* _mesh);

public:
    core::aabbox3df box;
    uint32_t meshBufCnt;
    uint64_t meshBufPtrs[1];
} PACK_STRUCT;
static_assert(sizeof(core::aabbox3df)==24, "sizeof(core::aabbox3df) must be 24");
static_assert(sizeof(MeshBlobV0::meshBufPtrs)==8, "sizeof(MeshBlobV0::meshBufPtrs) must be 8");
static_assert(
    sizeof(MeshBlobV0) ==
    sizeof(MeshBlobV0::box) + sizeof(MeshBlobV0::meshBufCnt) + sizeof(MeshBlobV0::meshBufPtrs),
    "MeshBlobV0: Size of blob is not sum of its contents!"
);
#include "irr/irrunpack.h"

using MeshBlobV1 = MeshBlobV0;
using MeshBlobV2 = MeshBlobV1;

template<>
struct CorrespondingBlobTypeFor<ICPUMesh> { typedef MeshBlobV2 type; };

}
} // irr::asset

#endif
