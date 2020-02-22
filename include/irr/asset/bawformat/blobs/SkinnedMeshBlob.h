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
struct IRR_FORCE_EBO SkinnedMeshBlobV3 : VariableSizeBlob<SkinnedMeshBlobV3,ICPUSkinnedMesh>, TypedBlob<SkinnedMeshBlobV3, ICPUSkinnedMesh>
{
public:
	enum E_BLOB_MESH_FLAG : uint32_t
	{
		EBMF_RIGHT_HANDED=0x1u
	};

    explicit SkinnedMeshBlobV3(const ICPUSkinnedMesh* _sm);

public:
    uint64_t boneHierarchyPtr;
    core::aabbox3df box;
	uint32_t meshFlags; // 1 bit used only for EBMF_RIGHT_HANDED
    uint32_t meshBufCnt;
    uint64_t meshBufPtrs[1];
} PACK_STRUCT;
static_assert(sizeof(SkinnedMeshBlobV3::meshBufPtrs)==8, "sizeof(SkinnedMeshBlobV0::meshBufPtrs) must be 8");
static_assert(
    sizeof(SkinnedMeshBlobV3) ==
    sizeof(SkinnedMeshBlobV3::boneHierarchyPtr) + sizeof(SkinnedMeshBlobV3::meshFlags) + sizeof(SkinnedMeshBlobV3::box) + sizeof(SkinnedMeshBlobV3::meshBufCnt) + sizeof(SkinnedMeshBlobV3::meshBufPtrs),
    "SkinnedMeshBlobV0: Size of blob is not sum of its contents!"
);
#include "irr/irrunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMesh> { typedef SkinnedMeshBlobV3 type; };

}
} // irr::asset

#endif
