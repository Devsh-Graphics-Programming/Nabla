// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_SKINNED_MESH_BUFFER_BLOB_H_INCLUDED__
#define __IRR_SKINNED_MESH_BUFFER_BLOB_H_INCLUDED__

namespace irr
{
namespace asset
{

class ICPUSkinnedMeshBuffer;

#include "irr/irrpack.h"
struct IRR_FORCE_EBO SkinnedMeshBufferBlobV3 : TypedBlob<SkinnedMeshBufferBlobV3, ICPUSkinnedMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV3, ICPUSkinnedMeshBuffer>
{
	//! Constructor filling all members
	explicit SkinnedMeshBufferBlobV3(const ICPUSkinnedMeshBuffer*);

	video::SCPUMaterial mat;
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
static_assert(sizeof(SkinnedMeshBufferBlobV3::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");
static_assert(
    sizeof(SkinnedMeshBufferBlobV3) ==
    sizeof(SkinnedMeshBufferBlobV3::mat) + sizeof(SkinnedMeshBufferBlobV3::box) + sizeof(SkinnedMeshBufferBlobV3::descPtr) + sizeof(SkinnedMeshBufferBlobV3::indexType) + sizeof(SkinnedMeshBufferBlobV3::baseVertex)
    + sizeof(SkinnedMeshBufferBlobV3::indexCount) + sizeof(SkinnedMeshBufferBlobV3::indexBufOffset) + sizeof(SkinnedMeshBufferBlobV3::instanceCount) + sizeof(SkinnedMeshBufferBlobV3::baseInstance)
    + sizeof(SkinnedMeshBufferBlobV3::primitiveType) + sizeof(SkinnedMeshBufferBlobV3::posAttrId) + sizeof(SkinnedMeshBufferBlobV3::normalAttrId) + sizeof(SkinnedMeshBufferBlobV3::indexValMin) + sizeof(SkinnedMeshBufferBlobV3::indexValMax) + sizeof(SkinnedMeshBufferBlobV3::maxVertexBoneInfluences),
    "SkinnedMeshBufferBlobV0: Size of blob is not sum of its contents!"
);
#include "irr/irrunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMeshBuffer> { typedef SkinnedMeshBufferBlobV3 type; };


}
} // irr::asset

#endif
