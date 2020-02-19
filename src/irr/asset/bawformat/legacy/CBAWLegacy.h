#ifndef __IRR_CBAW_LEGACY_H_INCLUDED__
#define __IRR_CBAW_LEGACY_H_INCLUDED__

#include "irr/asset/format/EFormat.h"

namespace irr
{
namespace asset
{


// forward declarations
class CFinalBoneHierarchy;


namespace legacyv0
{


enum E_COMPONENTS_PER_ATTRIBUTE
{
    //! Special ID for reverse XYZW order
    ECPA_REVERSED_OR_BGRA = 0,
    ECPA_ONE,
    ECPA_TWO,
    ECPA_THREE,
    ECPA_FOUR,
    ECPA_COUNT
};

enum E_COMPONENT_TYPE
{
    ECT_FLOAT = 0,
    ECT_HALF_FLOAT,
    ECT_DOUBLE_IN_FLOAT_OUT,
    ECT_UNSIGNED_INT_10F_11F_11F_REV,
    //INTEGER FORMS
    ECT_NORMALIZED_INT_2_10_10_10_REV,
    ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV,
    ECT_NORMALIZED_BYTE,
    ECT_NORMALIZED_UNSIGNED_BYTE,
    ECT_NORMALIZED_SHORT,
    ECT_NORMALIZED_UNSIGNED_SHORT,
    ECT_NORMALIZED_INT,
    ECT_NORMALIZED_UNSIGNED_INT,
    ECT_INT_2_10_10_10_REV,
    ECT_UNSIGNED_INT_2_10_10_10_REV,
    ECT_BYTE,
    ECT_UNSIGNED_BYTE,
    ECT_SHORT,
    ECT_UNSIGNED_SHORT,
    ECT_INT,
    ECT_UNSIGNED_INT,
    ECT_INTEGER_INT_2_10_10_10_REV,
    ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV,
    ECT_INTEGER_BYTE,
    ECT_INTEGER_UNSIGNED_BYTE,
    ECT_INTEGER_SHORT,
    ECT_INTEGER_UNSIGNED_SHORT,
    ECT_INTEGER_INT,
    ECT_INTEGER_UNSIGNED_INT,
    //special
    ECT_DOUBLE_IN_DOUBLE_OUT, //only accepted by glVertexAttribLPointer
    ECT_COUNT
};

#include "irr/irrpack.h"
//! Simple struct of essential data of ICPUMeshDataFormatDesc that has to be exported
//! Irrelevant in version 1.
//! @see @ref MeshDataFormatDescBlobV1
struct IRR_FORCE_EBO MeshDataFormatDescBlobV0 : TypedBlob<MeshDataFormatDescBlobV0, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >, VariableSizeBlob<MeshDataFormatDescBlobV0, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >
{
private:
    enum { VERTEX_ATTRIB_CNT = 16 };
public:
    //! was storing scene::E_COMPONENTS_PER_ATTRIBUTE in .baw v0 (in version 1 MeshDataFormatDescBlobV1 is used)
    uint32_t cpa[VERTEX_ATTRIB_CNT];
    //! was storing E_COMPONENT_TYPE in .baw v0
    uint32_t attrType[VERTEX_ATTRIB_CNT];
    size_t attrStride[VERTEX_ATTRIB_CNT];
    size_t attrOffset[VERTEX_ATTRIB_CNT];
    uint32_t attrDivisor[VERTEX_ATTRIB_CNT];
    uint64_t attrBufPtrs[VERTEX_ATTRIB_CNT];
    uint64_t idxBufPtr;
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(
    sizeof(MeshDataFormatDescBlobV0) == 
    sizeof(MeshDataFormatDescBlobV0::cpa) + sizeof(MeshDataFormatDescBlobV0::attrType) + sizeof(MeshDataFormatDescBlobV0::attrStride) + sizeof(MeshDataFormatDescBlobV0::attrOffset) + sizeof(MeshDataFormatDescBlobV0::attrDivisor) + sizeof(MeshDataFormatDescBlobV0::attrBufPtrs) + sizeof(MeshDataFormatDescBlobV0::idxBufPtr),
    "MeshDataFormatDescBlobV0: Size of blob is not sum of its contents!"
);

asset::E_FORMAT mapECT_plus_ECPA_onto_E_FORMAT(E_COMPONENT_TYPE _ct, E_COMPONENTS_PER_ATTRIBUTE _cpa);


#include "irr/irrpack.h"
struct IRR_FORCE_EBO FinalBoneHierarchyBlobV0 : VariableSizeBlob<FinalBoneHierarchyBlobV0,CFinalBoneHierarchy>, TypedBlob<FinalBoneHierarchyBlobV0, CFinalBoneHierarchy>
{
public:
	inline uint8_t* getBoneData()
	{
		return reinterpret_cast<uint8_t*>(this)+sizeof(FinalBoneHierarchyBlobV0);
	}

    size_t boneCount;
    size_t numLevelsInHierarchy;
    size_t keyframeCount;
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(
    sizeof(FinalBoneHierarchyBlobV0) ==
    sizeof(FinalBoneHierarchyBlobV0::boneCount) + sizeof(FinalBoneHierarchyBlobV0::numLevelsInHierarchy) + sizeof(FinalBoneHierarchyBlobV0::keyframeCount),
    "FinalBoneHierarchyBlobV0: Size of blob is not sum of its contents!"
);

class ICPUMesh;

#include "irr/irrpack.h"
//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
struct IRR_FORCE_EBO MeshBlobV0 : VariableSizeBlob<MeshBlobV0, asset::ICPUMesh>, TypedBlob<MeshBlobV0, asset::ICPUMesh>
{
public:
	core::aabbox3df box;
	uint32_t meshBufCnt;
	uint64_t meshBufPtrs[1];
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(sizeof(core::aabbox3df) == 24, "sizeof(core::aabbox3df) must be 24");
static_assert(sizeof(MeshBlobV0::meshBufPtrs) == 8, "sizeof(MeshBlobV0::meshBufPtrs) must be 8");
static_assert(
	sizeof(MeshBlobV0) ==
	sizeof(MeshBlobV0::box) + sizeof(MeshBlobV0::meshBufCnt) + sizeof(MeshBlobV0::meshBufPtrs),
	"MeshBlobV0: Size of blob is not sum of its contents!"
	);

class ICPUSkinnedMesh;

#include "irr/irrpack.h"
//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
struct IRR_FORCE_EBO SkinnedMeshBlobV0 : VariableSizeBlob<SkinnedMeshBlobV0, ICPUSkinnedMesh>, TypedBlob<SkinnedMeshBlobV0, ICPUSkinnedMesh>
{
public:
	uint64_t boneHierarchyPtr;
	core::aabbox3df box;
	uint32_t meshBufCnt;
	uint64_t meshBufPtrs[1];
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(sizeof(SkinnedMeshBlobV0::meshBufPtrs) == 8, "sizeof(SkinnedMeshBlobV0::meshBufPtrs) must be 8");
static_assert(
	sizeof(SkinnedMeshBlobV0) ==
	sizeof(SkinnedMeshBlobV0::boneHierarchyPtr) + sizeof(SkinnedMeshBlobV0::box) + sizeof(SkinnedMeshBlobV0::meshBufCnt) + sizeof(SkinnedMeshBlobV0::meshBufPtrs),
	"SkinnedMeshBlobV0: Size of blob is not sum of its contents!"
	);

#include "irr/irrpack.h"
//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
struct IRR_FORCE_EBO MeshBufferBlobV0 : TypedBlob<MeshBufferBlobV0, ICPUMeshBuffer>, FixedSizeBlob<MeshBufferBlobV0, ICPUMeshBuffer>
{
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
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(sizeof(MeshBufferBlobV0::mat) == 197, "sizeof(MeshBufferBlobV0::mat) must be 197");
static_assert(
	sizeof(MeshBufferBlobV0) ==
	sizeof(MeshBufferBlobV0::mat) + sizeof(MeshBufferBlobV0::box) + sizeof(MeshBufferBlobV0::descPtr) + sizeof(MeshBufferBlobV0::indexType) + sizeof(MeshBufferBlobV0::baseVertex)
	+ sizeof(MeshBufferBlobV0::indexCount) + sizeof(MeshBufferBlobV0::indexBufOffset) + sizeof(MeshBufferBlobV0::instanceCount) + sizeof(MeshBufferBlobV0::baseInstance)
	+ sizeof(MeshBufferBlobV0::primitiveType) + sizeof(MeshBufferBlobV0::posAttrId),
	"MeshBufferBlobV0: Size of blob is not sum of its contents!"
	);

class ICPUSkinnedMeshBuffer;

#include "irr/irrpack.h"
struct IRR_FORCE_EBO SkinnedMeshBufferBlobV0 : TypedBlob<SkinnedMeshBufferBlobV0, ICPUSkinnedMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV0, ICPUSkinnedMeshBuffer>
{
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
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(sizeof(SkinnedMeshBufferBlobV0::mat) == 197, "sizeof(MeshBufferBlobV0::mat) must be 197");
static_assert(
	sizeof(SkinnedMeshBufferBlobV0) ==
	sizeof(SkinnedMeshBufferBlobV0::mat) + sizeof(SkinnedMeshBufferBlobV0::box) + sizeof(SkinnedMeshBufferBlobV0::descPtr) + sizeof(SkinnedMeshBufferBlobV0::indexType) + sizeof(SkinnedMeshBufferBlobV0::baseVertex)
	+ sizeof(SkinnedMeshBufferBlobV0::indexCount) + sizeof(SkinnedMeshBufferBlobV0::indexBufOffset) + sizeof(SkinnedMeshBufferBlobV0::instanceCount) + sizeof(SkinnedMeshBufferBlobV0::baseInstance)
	+ sizeof(SkinnedMeshBufferBlobV0::primitiveType) + sizeof(SkinnedMeshBufferBlobV0::posAttrId) + sizeof(SkinnedMeshBufferBlobV0::indexValMin) + sizeof(SkinnedMeshBufferBlobV0::indexValMax) + sizeof(SkinnedMeshBufferBlobV0::maxVertexBoneInfluences),
	"SkinnedMeshBufferBlobV0: Size of blob is not sum of its contents!"
	);

}


namespace legacyv1
{
	
using FinalBoneHierarchyBlobV1 = legacyv0::FinalBoneHierarchyBlobV0;
using MeshBlobV1 = legacyv0::MeshBlobV0;
using SkinnedMeshBlobV1 = legacyv0::SkinnedMeshBlobV0;
using MeshBufferBlobV1 = legacyv0::MeshBufferBlobV0;
using SkinnedMeshBufferBlobV1 = legacyv0::SkinnedMeshBufferBlobV0;

}

namespace legacyv2
{

#include "irr/irrpack.h"
struct IRR_FORCE_EBO FinalBoneHierarchyBlobV2 : VariableSizeBlob<FinalBoneHierarchyBlobV2, CFinalBoneHierarchy>, TypedBlob<FinalBoneHierarchyBlobV2, CFinalBoneHierarchy>
{
public:

	size_t boneCount;
	size_t numLevelsInHierarchy;
	size_t keyframeCount;
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(
	sizeof(FinalBoneHierarchyBlobV2) ==
	sizeof(FinalBoneHierarchyBlobV2::boneCount) + sizeof(FinalBoneHierarchyBlobV2::numLevelsInHierarchy) + sizeof(FinalBoneHierarchyBlobV2::keyframeCount),
	"FinalBoneHierarchyBlobV2: Size of blob is not sum of its contents!"
	);


using MeshBlobV2 = legacyv1::MeshBlobV1;
using SkinnedMeshBlobV2 = legacyv1::SkinnedMeshBlobV1;
using MeshBufferBlobV2 = legacyv1::MeshBufferBlobV1;
using SkinnedMeshBufferBlobV2 = legacyv1::SkinnedMeshBufferBlobV1;

}

}
} //irr::asset

#endif //__IRR_CBAW_LEGACY_H_INCLUDED__