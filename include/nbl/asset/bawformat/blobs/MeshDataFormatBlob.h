// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_MESH_DATA_FORMAT_DESC_BLOB_H_INCLUDED__
#define __NBL_ASSET_MESH_DATA_FORMAT_DESC_BLOB_H_INCLUDED__

#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{
//template<typename> class IMeshDataFormatDesc; // is this the type we should be using?

namespace legacyv0
{
struct MeshDataFormatDescBlobV0;
}

#ifdef OLD_SHADERS
#include "nbl/nblpack.h"
struct NBL_FORCE_EBO MeshDataFormatDescBlobV1 : TypedBlob<MeshDataFormatDescBlobV1, IMeshDataFormatDesc<ICPUBuffer> >, FixedSizeBlob<MeshDataFormatDescBlobV1, IMeshDataFormatDesc<ICPUBuffer> >
{
private:
    enum
    {
        VERTEX_ATTRIB_CNT = 16
    };

public:
    //! Constructor filling all members
    explicit MeshDataFormatDescBlobV1(const IMeshDataFormatDesc<ICPUBuffer>*);
    //! Backward compatibility constructor
    explicit MeshDataFormatDescBlobV1(const legacyv0::MeshDataFormatDescBlobV0&);

    uint32_t attrFormat[VERTEX_ATTRIB_CNT];
    uint32_t attrStride[VERTEX_ATTRIB_CNT];
    size_t attrOffset[VERTEX_ATTRIB_CNT];
    uint32_t attrDivisor;
    uint32_t padding;
    uint64_t attrBufPtrs[VERTEX_ATTRIB_CNT];
    uint64_t idxBufPtr;
} PACK_STRUCT;
#include "nbl/nblunpack.h"
static_assert(
    sizeof(MeshDataFormatDescBlobV1) ==
        sizeof(MeshDataFormatDescBlobV1::attrFormat) + sizeof(MeshDataFormatDescBlobV1::attrStride) + sizeof(MeshDataFormatDescBlobV1::attrOffset) + sizeof(MeshDataFormatDescBlobV1::attrDivisor) + sizeof(MeshDataFormatDescBlobV1::padding) + sizeof(MeshDataFormatDescBlobV1::attrBufPtrs) + sizeof(MeshDataFormatDescBlobV1::idxBufPtr),
    "MeshDataFormatDescBlobV1: Size of blob is not sum of its contents!");

using MeshDataFormatDescBlobV2 = MeshDataFormatDescBlobV1;
using MeshDataFormatDescBlobV3 = MeshDataFormatDescBlobV2;

template<>
struct CorrespondingBlobTypeFor<IMeshDataFormatDesc<ICPUBuffer> >
{
    typedef MeshDataFormatDescBlobV3 type;
};

template<>
inline size_t SizedBlob<FixedSizeBlob, MeshDataFormatDescBlobV3, IMeshDataFormatDesc<ICPUBuffer> >::calcBlobSizeForObj(const IMeshDataFormatDesc<ICPUBuffer>* _obj)
{
    return sizeof(MeshDataFormatDescBlobV3);
}
#endif

}
}  // nbl::asset

#endif
