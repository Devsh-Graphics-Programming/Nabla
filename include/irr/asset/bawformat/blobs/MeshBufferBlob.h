// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_MESH_BUFFER_BLOB_H_INCLUDED__
#define __IRR_MESH_BUFFER_BLOB_H_INCLUDED__


namespace irr
{
namespace asset
{

class ICPUMeshBuffer;

#include "irr/irrpack.h"
//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
struct IRR_FORCE_EBO MeshBufferBlobV0 : TypedBlob<MeshBufferBlobV0, ICPUMeshBuffer>, FixedSizeBlob<MeshBufferBlobV0, ICPUMeshBuffer>
{
	//! Constructor filling all members
	explicit MeshBufferBlobV0(const ICPUMeshBuffer*);

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
} PACK_STRUCT;
#include "irr/irrunpack.h"
//TODO bring it back
//static_assert(sizeof(MeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");

#ifndef NEW_SHADERS
static_assert(
    sizeof(MeshBufferBlobV0) ==
    sizeof(MeshBufferBlobV0::mat) + sizeof(MeshBufferBlobV0::box) + sizeof(MeshBufferBlobV0::descPtr) + sizeof(MeshBufferBlobV0::indexType) + sizeof(MeshBufferBlobV0::baseVertex)
    + sizeof(MeshBufferBlobV0::indexCount) + sizeof(MeshBufferBlobV0::indexBufOffset) + sizeof(MeshBufferBlobV0::instanceCount) + sizeof(MeshBufferBlobV0::baseInstance)
    + sizeof(MeshBufferBlobV0::primitiveType) + sizeof(MeshBufferBlobV0::posAttrId),
    "MeshBufferBlobV0: Size of blob is not sum of its contents!"
);
#endif

using MeshBufferBlobV1 = MeshBufferBlobV0;
using MeshBufferBlobV2 = MeshBufferBlobV1;


template<>
struct CorrespondingBlobTypeFor<ICPUMeshBuffer> { typedef MeshBufferBlobV2 type; };

}
} // irr::asset

#endif
