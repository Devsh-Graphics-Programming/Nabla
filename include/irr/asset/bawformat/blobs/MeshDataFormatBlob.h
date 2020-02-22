// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_MESH_DATA_FORMAT_DESC_BLOB_H_INCLUDED__
#define __IRR_MESH_DATA_FORMAT_DESC_BLOB_H_INCLUDED__

#include "irr/asset/ICPUBuffer.h"

namespace irr
{
namespace asset
{

template<typename> class IMeshDataFormatDesc; // is this the type we should be using?

namespace legacyv0
{
	struct MeshDataFormatDescBlobV0;
}

#include "irr/irrpack.h"
struct IRR_FORCE_EBO MeshDataFormatDescBlobV1 : TypedBlob<MeshDataFormatDescBlobV1, IMeshDataFormatDesc<ICPUBuffer> >, FixedSizeBlob<MeshDataFormatDescBlobV1, IMeshDataFormatDesc<ICPUBuffer> >
{
private:
    enum { VERTEX_ATTRIB_CNT = 16 };
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
#include "irr/irrunpack.h"
static_assert(
    sizeof(MeshDataFormatDescBlobV1) ==
    sizeof(MeshDataFormatDescBlobV1::attrFormat) + sizeof(MeshDataFormatDescBlobV1::attrStride) + sizeof(MeshDataFormatDescBlobV1::attrOffset) + sizeof(MeshDataFormatDescBlobV1::attrDivisor) + sizeof(MeshDataFormatDescBlobV1::padding) + sizeof(MeshDataFormatDescBlobV1::attrBufPtrs) + sizeof(MeshDataFormatDescBlobV1::idxBufPtr),
    "MeshDataFormatDescBlobV1: Size of blob is not sum of its contents!"
);

using MeshDataFormatDescBlobV2 = MeshDataFormatDescBlobV1;
using MeshDataFormatDescBlobV3 = MeshDataFormatDescBlobV2;

template<>
struct CorrespondingBlobTypeFor<IMeshDataFormatDesc<ICPUBuffer> > { typedef MeshDataFormatDescBlobV3 type; };

template<>
inline size_t SizedBlob<FixedSizeBlob, MeshDataFormatDescBlobV3, IMeshDataFormatDesc<ICPUBuffer> >::calcBlobSizeForObj(const IMeshDataFormatDesc<ICPUBuffer>* _obj)
{
    return sizeof(MeshDataFormatDescBlobV3);
}


}
} // irr::asset

#endif
