// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MESH_BUFFER_BLOB_H_INCLUDED__
#define __NBL_ASSET_MESH_BUFFER_BLOB_H_INCLUDED__

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/bawformat/Blob.h"

namespace nbl
{
namespace asset
{
class ICPUMeshBuffer;

#include "nbl/nblpack.h"
//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
struct NBL_FORCE_EBO MeshBufferBlobV3 : TypedBlob<MeshBufferBlobV3, ICPUMeshBuffer>, FixedSizeBlob<MeshBufferBlobV3, ICPUMeshBuffer>
{
    //! Constructor filling all members
    explicit MeshBufferBlobV3(const ICPUMeshBuffer*);

#ifdef OLD_SHADERS
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
    uint32_t normalAttrId;
} PACK_STRUCT;
#include "nbl/nblunpack.h"
//TODO bring it back
//static_assert(sizeof(MeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");

#ifdef OLD_SHADERS
static_assert(
    sizeof(MeshBufferBlobV3) ==
        sizeof(MeshBufferBlobV3::mat) + sizeof(MeshBufferBlobV3::box) + sizeof(MeshBufferBlobV3::descPtr) + sizeof(MeshBufferBlobV3::indexType) + sizeof(MeshBufferBlobV3::baseVertex) + sizeof(MeshBufferBlobV3::indexCount) + sizeof(MeshBufferBlobV3::indexBufOffset) + sizeof(MeshBufferBlobV3::instanceCount) + sizeof(MeshBufferBlobV3::baseInstance) + sizeof(MeshBufferBlobV3::primitiveType) + sizeof(MeshBufferBlobV3::posAttrId) + sizeof(MeshBufferBlobV3::normalAttrId),
    "MeshBufferBlobV0: Size of blob is not sum of its contents!");
#endif

template<>
struct CorrespondingBlobTypeFor<ICPUMeshBuffer>
{
    typedef MeshBufferBlobV3 type;
};

}
}  // nbl::asset

#endif
