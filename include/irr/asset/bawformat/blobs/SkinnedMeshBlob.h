// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_SKINNED_MESH_BLOB_H_INCLUDED__
#define __IRR_SKINNED_MESH_BLOB_H_INCLUDED__

namespace irr
{
namespace asset
{

class ICPUSkinnedMesh;

#include "irr/irrpack.h"
//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
struct IRR_FORCE_EBO SkinnedMeshBlobV0 : VariableSizeBlob<SkinnedMeshBlobV0,ICPUSkinnedMesh>, TypedBlob<SkinnedMeshBlobV0, ICPUSkinnedMesh>
{
	//friend struct SizedBlob<VariableSizeBlob, SkinnedMeshBlobV0, ICPUSkinnedMesh>;
public:
    explicit SkinnedMeshBlobV0(const ICPUSkinnedMesh* _sm);

public:
    uint64_t boneHierarchyPtr;
    core::aabbox3df box;
    uint32_t meshBufCnt;
    uint64_t meshBufPtrs[1];
} PACK_STRUCT;
static_assert(sizeof(SkinnedMeshBlobV0::meshBufPtrs)==8, "sizeof(SkinnedMeshBlobV0::meshBufPtrs) must be 8");
static_assert(
    sizeof(SkinnedMeshBlobV0) ==
    sizeof(SkinnedMeshBlobV0::boneHierarchyPtr) + sizeof(SkinnedMeshBlobV0::box) + sizeof(SkinnedMeshBlobV0::meshBufCnt) + sizeof(SkinnedMeshBlobV0::meshBufPtrs),
    "SkinnedMeshBlobV0: Size of blob is not sum of its contents!"
);
#include "irr/irrunpack.h"

using SkinnedMeshBlobV1 = SkinnedMeshBlobV0;
using SkinnedMeshBlobV2 = SkinnedMeshBlobV1;

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMesh> { typedef SkinnedMeshBlobV2 type; };

}
} // irr::asset

#endif
