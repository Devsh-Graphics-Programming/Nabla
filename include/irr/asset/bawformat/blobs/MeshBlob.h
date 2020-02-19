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
struct IRR_FORCE_EBO MeshBlobV3 : VariableSizeBlob<MeshBlobV3,asset::ICPUMesh>, TypedBlob<MeshBlobV3, asset::ICPUMesh>
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
#include "irr/irrunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUMesh> { typedef MeshBlobV3 type; };

}
} // irr::asset

#endif
