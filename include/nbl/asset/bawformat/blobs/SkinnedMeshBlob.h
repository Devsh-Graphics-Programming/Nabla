// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_SKINNED_MESH_BLOB_H_INCLUDED__
#define __NBL_ASSET_SKINNED_MESH_BLOB_H_INCLUDED__

namespace nbl
{
namespace asset
{
class ICPUSkinnedMesh;

#include "nbl/nblpack.h"
//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
struct NBL_FORCE_EBO SkinnedMeshBlobV3 : VariableSizeBlob<SkinnedMeshBlobV3, ICPUSkinnedMesh>, TypedBlob<SkinnedMeshBlobV3, ICPUSkinnedMesh>
{
public:
    enum E_BLOB_MESH_FLAG : uint32_t
    {
        EBMF_RIGHT_HANDED = 0x1u
    };

    explicit SkinnedMeshBlobV3(const ICPUSkinnedMesh* _sm);

public:
    uint64_t boneHierarchyPtr;
    core::aabbox3df box;
    uint32_t meshFlags;  // 1 bit used only for EBMF_RIGHT_HANDED
    uint32_t meshBufCnt;
    uint64_t meshBufPtrs[1];
} PACK_STRUCT;
static_assert(sizeof(SkinnedMeshBlobV3::meshBufPtrs) == 8, "sizeof(SkinnedMeshBlobV0::meshBufPtrs) must be 8");
static_assert(
    sizeof(SkinnedMeshBlobV3) ==
        sizeof(SkinnedMeshBlobV3::boneHierarchyPtr) + sizeof(SkinnedMeshBlobV3::meshFlags) + sizeof(SkinnedMeshBlobV3::box) + sizeof(SkinnedMeshBlobV3::meshBufCnt) + sizeof(SkinnedMeshBlobV3::meshBufPtrs),
    "SkinnedMeshBlobV0: Size of blob is not sum of its contents!");
#include "nbl/nblunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMesh>
{
    typedef SkinnedMeshBlobV3 type;
};

}
}  // nbl::asset

#endif
