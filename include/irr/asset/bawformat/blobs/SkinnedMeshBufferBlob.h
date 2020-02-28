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
struct IRR_FORCE_EBO SkinnedMeshBufferBlobV0 : TypedBlob<SkinnedMeshBufferBlobV0, ICPUSkinnedMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV0, ICPUSkinnedMeshBuffer>
{
	//! Constructor filling all members
	explicit SkinnedMeshBufferBlobV0(const ICPUSkinnedMeshBuffer*);

#ifndef NEW_SHADERS
	video::SCPUMaterial mat;
#endif
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
} PACK_STRUCT;
//TODO bring it back
//static_assert(sizeof(SkinnedMeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");

#ifndef NEW_SHADERS
static_assert(
    sizeof(SkinnedMeshBufferBlobV0) ==
    sizeof(SkinnedMeshBufferBlobV0::mat) + sizeof(SkinnedMeshBufferBlobV0::box) + sizeof(SkinnedMeshBufferBlobV0::descPtr) + sizeof(SkinnedMeshBufferBlobV0::indexType) + sizeof(SkinnedMeshBufferBlobV0::baseVertex)
    + sizeof(SkinnedMeshBufferBlobV0::indexCount) + sizeof(SkinnedMeshBufferBlobV0::indexBufOffset) + sizeof(SkinnedMeshBufferBlobV0::instanceCount) + sizeof(SkinnedMeshBufferBlobV0::baseInstance)
    + sizeof(SkinnedMeshBufferBlobV0::primitiveType) + sizeof(SkinnedMeshBufferBlobV0::posAttrId) + sizeof(SkinnedMeshBufferBlobV0::indexValMin) + sizeof(SkinnedMeshBufferBlobV0::indexValMax) + sizeof(SkinnedMeshBufferBlobV0::maxVertexBoneInfluences),
    "SkinnedMeshBufferBlobV0: Size of blob is not sum of its contents!"
);
#endif
#include "irr/irrunpack.h"

using SkinnedMeshBufferBlobV1 = SkinnedMeshBufferBlobV0;
using SkinnedMeshBufferBlobV2 = SkinnedMeshBufferBlobV1;

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMeshBuffer> { typedef SkinnedMeshBufferBlobV2 type; };


}
} // irr::asset

#endif
