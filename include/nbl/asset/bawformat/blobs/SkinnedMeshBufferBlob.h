// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_SKINNED_MESH_BUFFER_BLOB_H_INCLUDED__
#define __NBL_ASSET_SKINNED_MESH_BUFFER_BLOB_H_INCLUDED__

namespace nbl
{
namespace asset
{
#ifdef OLD_SHADERS
class ICPUSkinnedMeshBuffer;

#include "nbl/nblpack.h"
struct NBL_FORCE_EBO SkinnedMeshBufferBlobV3 : TypedBlob<SkinnedMeshBufferBlobV3, ICPUSkinnedMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV3, ICPUSkinnedMeshBuffer>
{
    //! Constructor filling all members
    explicit SkinnedMeshBufferBlobV3(const ICPUSkinnedMeshBuffer*);

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
    uint32_t indexValMin;
    uint32_t indexValMax;
    uint32_t maxVertexBoneInfluences;
    uint32_t normalAttrId;
} PACK_STRUCT;
//TODO bring it back
//static_assert(sizeof(SkinnedMeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");

static_assert(
    sizeof(SkinnedMeshBufferBlobV3) ==
        sizeof(SkinnedMeshBufferBlobV3::mat) + sizeof(SkinnedMeshBufferBlobV3::box) + sizeof(SkinnedMeshBufferBlobV3::descPtr) + sizeof(SkinnedMeshBufferBlobV3::indexType) + sizeof(SkinnedMeshBufferBlobV3::baseVertex) + sizeof(SkinnedMeshBufferBlobV3::indexCount) + sizeof(SkinnedMeshBufferBlobV3::indexBufOffset) + sizeof(SkinnedMeshBufferBlobV3::instanceCount) + sizeof(SkinnedMeshBufferBlobV3::baseInstance) + sizeof(SkinnedMeshBufferBlobV3::primitiveType) + sizeof(SkinnedMeshBufferBlobV3::posAttrId) + sizeof(SkinnedMeshBufferBlobV3::normalAttrId) + sizeof(SkinnedMeshBufferBlobV3::indexValMin) + sizeof(SkinnedMeshBufferBlobV3::indexValMax) + sizeof(SkinnedMeshBufferBlobV3::maxVertexBoneInfluences),
    "SkinnedMeshBufferBlobV0: Size of blob is not sum of its contents!");
#include "nbl/nblunpack.h"

template<>
struct CorrespondingBlobTypeFor<ICPUSkinnedMeshBuffer>
{
    typedef SkinnedMeshBufferBlobV3 type;
};
#endif

}
}  // nbl::asset

#endif
