// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_SKINNED_MESH_BUFFER_BLOB_H_INCLUDED__
#define __NBL_ASSET_SKINNED_MESH_BUFFER_BLOB_H_INCLUDED__

namespace irr
{
namespace asset
{

class ICPUSkinnedMeshBuffer;

#include "irr/irrpack.h"
struct NBL_FORCE_EBO SkinnedMeshBufferBlobV3 : TypedBlob<SkinnedMeshBufferBlobV3, ICPUSkinnedMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV3, ICPUSkinnedMeshBuffer>
{
	//! Constructor filling all members
	explicit SkinnedMeshBufferBlobV3(const ICPUSkinnedMeshBuffer*);

	core::aabbox3df box;
	uint64_t descPtr;
	uint32_t indexType;
	uint32_t baseVertex;
	uint64_t indexCount;
	size_t indexBufOffset;
	size_t instanceCount;
	uint32_t baseInstance;
	uint32_t primitiveType;
	uint32_t posAttrId;
	uint32_t indexValMin;
	uint32_t indexValMax;
	uint32_t maxVertexBoneInfluences;
	uint32_t normalAttrId;
} PACK_STRUCT;
//TODO bring it back
//static_assert(sizeof(SkinnedMeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");

#include "irr/irrunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMeshBuffer> { typedef SkinnedMeshBufferBlobV3 type; };


}
} // irr::asset

#endif
