// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#ifndef _NBL_ASSET_MESH_BLOB_H_INCLUDED_
#define _NBL_ASSET_MESH_BLOB_H_INCLUDED_

#include "nbl/asset/bawformat/Blob.h"

namespace nbl::asset
{

class ICPUMesh;

#if 0
#include "nbl/nblpack.h"
//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
struct NBL_FORCE_EBO MeshBlobV3 : VariableSizeBlob<MeshBlobV3,asset::ICPUMesh>, TypedBlob<MeshBlobV3, asset::ICPUMesh>
{
public:
	enum E_BLOB_MESH_FLAG : uint32_t
	{
		EBMF_RIGHT_HANDED = 0x1u
	};

	explicit MeshBlobV3(const asset::ICPUMesh* _mesh);

public:
    core::aabbox3df box;
	uint32_t meshFlags; // 1 bit used only for EBMF_RIGHT_HANDED
    uint32_t meshBufCnt;
    uint64_t meshBufPtrs[1];
} PACK_STRUCT;
static_assert(sizeof(core::aabbox3df)==24, "sizeof(core::aabbox3df) must be 24");
static_assert(sizeof(MeshBlobV3::meshBufPtrs)==8, "sizeof(MeshBlobV0::meshBufPtrs) must be 8");
static_assert(
    sizeof(MeshBlobV3) ==
    sizeof(MeshBlobV3::box) + sizeof(MeshBlobV3::meshBufCnt) + sizeof(MeshBlobV3::meshFlags) + sizeof(MeshBlobV3::meshBufPtrs),
    "MeshBlobV0: Size of blob is not sum of its contents!"
);
#include "nbl/nblunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUMesh> { typedef MeshBlobV3 type; };
#endif

} // nbl::asset

#endif
