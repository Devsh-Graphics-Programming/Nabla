// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_MESH_BUFFER_BLOB_H_INCLUDED__
#define __IRR_MESH_BUFFER_BLOB_H_INCLUDED__

#include "irr/asset/bawformat/legacy/CBAWLegacy.h"

namespace irr
{
namespace asset
{

class ICPUMeshBuffer;

#include "irr/irrpack.h"
//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
struct IRR_FORCE_EBO MeshBufferBlobV3 : TypedBlob<MeshBufferBlobV3, ICPUMeshBuffer>, FixedSizeBlob<MeshBufferBlobV3, ICPUMeshBuffer>
{
	//! Constructor filling all members
	explicit MeshBufferBlobV3(const ICPUMeshBuffer*);

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
	uint32_t normalAttrId;
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(sizeof(MeshBufferBlobV3::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");
static_assert(
    sizeof(MeshBufferBlobV3) ==
    sizeof(MeshBufferBlobV3::mat) + sizeof(MeshBufferBlobV3::box) + sizeof(MeshBufferBlobV3::descPtr) + sizeof(MeshBufferBlobV3::indexType) + sizeof(MeshBufferBlobV3::baseVertex)
    + sizeof(MeshBufferBlobV3::indexCount) + sizeof(MeshBufferBlobV3::indexBufOffset) + sizeof(MeshBufferBlobV3::instanceCount) + sizeof(MeshBufferBlobV3::baseInstance)
    + sizeof(MeshBufferBlobV3::primitiveType) + sizeof(MeshBufferBlobV3::posAttrId) + sizeof(MeshBufferBlobV3::normalAttrId),
    "MeshBufferBlobV0: Size of blob is not sum of its contents!"
);

template<>
struct CorrespondingBlobTypeFor<ICPUMeshBuffer> { typedef MeshBufferBlobV3 type; };

}
} // irr::asset

#endif
