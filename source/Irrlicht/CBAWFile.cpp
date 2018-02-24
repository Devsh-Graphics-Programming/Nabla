// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWFile.h"

#include "ISkinnedMesh.h"
#include "SSkinMeshBuffer.h"
#include "CFinalBoneHierarchy.h"
#include "coreutil.h"

//! for C++11
//using namespace std;

namespace irr { namespace core
{

void core::BlobHeaderV0::finalize(const void* _notCompressedData, const void* _data, size_t _sizeDecompr, size_t _sizeCompr, uint8_t _comprType)
{
	blobSizeDecompr = _sizeDecompr;
	blobSize = _sizeCompr;
	compressionType = _comprType;

	//compress before encrypting, increased entropy makes compression hard

	//compress and encrypt before hashing

	core::XXHash_256(_notCompressedData, blobSizeDecompr, blobHash);
}

bool core::BlobHeaderV0::validate(const void* _decomprData) const
{
    uint64_t tmpHash[4];
	core::XXHash_256(_decomprData, blobSizeDecompr, tmpHash);
	for (size_t i=0; i<4; i++)
		if (tmpHash[i] != blobHash[i])
			return false;
    return true;
}


MeshBlobV0::MeshBlobV0(const scene::ICPUMesh* _mesh) : box(_mesh->getBoundingBox()), meshBufCnt(_mesh->getMeshBufferCount())
{
	for (uint32_t i = 0; i < meshBufCnt; ++i)
		meshBufPtrs[i] = reinterpret_cast<uint64_t>(_mesh->getMeshBuffer(i));
}

template<>
size_t SizedBlob<VariableSizeBlob, MeshBlobV0, scene::ICPUMesh>::calcBlobSizeForObj(const scene::ICPUMesh* _obj)
{
	return sizeof(MeshBlobV0) + (_obj->getMeshBufferCount()-1) * sizeof(uint64_t);
}

SkinnedMeshBlobV0::SkinnedMeshBlobV0(const scene::ICPUSkinnedMesh* _sm)
	: boneHierarchyPtr(reinterpret_cast<uint64_t>(_sm->getBoneReferenceHierarchy())), box(_sm->getBoundingBox()), meshBufCnt(_sm->getMeshBufferCount())
{
	for (uint32_t i = 0; i < meshBufCnt; ++i)
	{
		meshBufPtrs[i] = reinterpret_cast<uint64_t>(_sm->getMeshBuffer(i));
	}
}

template<>
size_t SizedBlob<VariableSizeBlob, SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::calcBlobSizeForObj(const scene::ICPUSkinnedMesh* _obj)
{
	return sizeof(SkinnedMeshBlobV0) + (_obj->getMeshBufferCount() - 1) * sizeof(uint64_t);
}

MeshBufferBlobV0::MeshBufferBlobV0(const scene::ICPUMeshBuffer* _mb)
{
	memcpy(&mat, &_mb->getMaterial(), sizeof(video::SMaterial));
	memcpy(&box, &_mb->getBoundingBox(), sizeof(core::aabbox3df));
	descPtr = reinterpret_cast<uint64_t>(_mb->getMeshDataAndFormat());
	indexType = _mb->getIndexType();
	baseVertex = _mb->getBaseVertex();
	indexCount = _mb->getIndexCount();
	indexBufOffset = _mb->getIndexBufferOffset();
	instanceCount = _mb->getInstanceCount();
	baseInstance = _mb->getBaseInstance();
	primitiveType = _mb->getPrimitiveType();
	posAttrId = _mb->getPositionAttributeIx();
}

template<>
size_t SizedBlob<FixedSizeBlob, MeshBufferBlobV0, scene::ICPUMeshBuffer>::calcBlobSizeForObj(const scene::ICPUMeshBuffer* _obj)
{
	return sizeof(MeshBufferBlobV0);
}

SkinnedMeshBufferBlobV0::SkinnedMeshBufferBlobV0(const scene::SCPUSkinMeshBuffer* _smb)
{
	memcpy(&mat, &_smb->getMaterial(), sizeof(video::SMaterial));
	memcpy(&box, &_smb->getBoundingBox(), sizeof(core::aabbox3df));
	descPtr = reinterpret_cast<uint64_t>(_smb->getMeshDataAndFormat());
	indexType = _smb->getIndexType();
	baseVertex = _smb->getBaseVertex();
	indexCount = _smb->getIndexCount();
	indexBufOffset = _smb->getIndexBufferOffset();
	instanceCount = _smb->getInstanceCount();
	baseInstance = _smb->getBaseInstance();
	primitiveType = _smb->getPrimitiveType();
	posAttrId = _smb->getPositionAttributeIx();
	indexValMin = _smb->getIndexMinBound();
	indexValMax = _smb->getIndexMaxBound();
	maxVertexBoneInfluences = _smb->getMaxVertexBoneInfluences();
}

template<>
size_t SizedBlob<FixedSizeBlob, SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::calcBlobSizeForObj(const scene::SCPUSkinMeshBuffer* _obj)
{
	return sizeof(SkinnedMeshBufferBlobV0);
}

MeshDataFormatDescBlobV0::MeshDataFormatDescBlobV0(const scene::IMeshDataFormatDesc<core::ICPUBuffer>* _desc)
{
	using namespace scene;

	_IRR_DEBUG_BREAK_IF(VERTEX_ATTRIB_CNT != EVAI_COUNT) // could be static_assert if had c++11

	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
		cpa[(int)i] = _desc->getAttribComponentCount(i);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
		attrType[(int)i] = _desc->getAttribType(i);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
		attrStride[(int)i] = _desc->getMappedBufferStride(i);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
		attrOffset[(int)i] = _desc->getMappedBufferOffset(i);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
		attrDivisor[(int)i] = _desc->getAttribDivisor(i);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
		attrBufPtrs[(int)i] = reinterpret_cast<uint64_t>(_desc->getMappedBuffer(i));

	idxBufPtr = reinterpret_cast<uint64_t>(_desc->getIndexBuffer());
}

template<>
size_t SizedBlob<FixedSizeBlob, MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::calcBlobSizeForObj(const scene::IMeshDataFormatDesc<core::ICPUBuffer>* _obj)
{
	return sizeof(MeshDataFormatDescBlobV0);
}

FinalBoneHierarchyBlobV0::FinalBoneHierarchyBlobV0(const scene::CFinalBoneHierarchy* _fbh)
{
	boneCount = _fbh->getBoneCount();
	numLevelsInHierarchy = _fbh->getHierarchyLevels();
	keyframeCount = _fbh->getKeyFrameCount();

	uint8_t* const ptr = ((uint8_t*)this);
	memcpy(ptr + calcBonesOffset(_fbh), _fbh->getBoneData(), calcBonesByteSize(_fbh));
	memcpy(ptr + calcLevelsOffset(_fbh), _fbh->getBoneTreeLevelEnd(), calcLevelsByteSize(_fbh));
	memcpy(ptr + calcKeyFramesOffset(_fbh), _fbh->getKeys(), calcKeyFramesByteSize(_fbh));
	memcpy(ptr + calcInterpolatedAnimsOffset(_fbh), _fbh->getInterpolatedAnimationData(), calcInterpolatedAnimsByteSize(_fbh));
	memcpy(ptr + calcNonInterpolatedAnimsOffset(_fbh), _fbh->getNonInterpolatedAnimationData(), calcNonInterpolatedAnimsByteSize(_fbh));
	uint8_t* strPtr = ptr + calcBoneNamesOffset(_fbh);
	for (size_t i = 0; i < boneCount; ++i)
	{
		memcpy(strPtr, _fbh->getBoneName(i).c_str(), _fbh->getBoneName(i).size());
		strPtr += _fbh->getBoneName(i).size();
		*strPtr = 0;
		++strPtr;
	}
}

template<>
size_t SizedBlob<VariableSizeBlob, FinalBoneHierarchyBlobV0,scene::CFinalBoneHierarchy>::calcBlobSizeForObj(const scene::CFinalBoneHierarchy* _obj)
{
	return
		sizeof(FinalBoneHierarchyBlobV0) +
		FinalBoneHierarchyBlobV0::calcBonesByteSize(_obj) +
		FinalBoneHierarchyBlobV0::calcLevelsByteSize(_obj) +
		FinalBoneHierarchyBlobV0::calcKeyFramesByteSize(_obj) +
		FinalBoneHierarchyBlobV0::calcInterpolatedAnimsByteSize(_obj) +
		FinalBoneHierarchyBlobV0::calcNonInterpolatedAnimsByteSize(_obj) +
		FinalBoneHierarchyBlobV0::calcBoneNamesByteSize(_obj);
}

size_t FinalBoneHierarchyBlobV0::calcBonesOffset(const scene::CFinalBoneHierarchy* _fbh)
{
	return sizeof(FinalBoneHierarchyBlobV0);
}
size_t FinalBoneHierarchyBlobV0::calcLevelsOffset(const scene::CFinalBoneHierarchy * _fbh)
{
	return calcBonesOffset(_fbh) + calcBonesByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV0::calcKeyFramesOffset(const scene::CFinalBoneHierarchy * _fbh)
{
	return calcLevelsOffset(_fbh) + calcLevelsByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV0::calcInterpolatedAnimsOffset(const scene::CFinalBoneHierarchy * _fbh)
{
	return calcKeyFramesOffset(_fbh) + calcKeyFramesByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV0::calcNonInterpolatedAnimsOffset(const scene::CFinalBoneHierarchy * _fbh)
{
	return calcInterpolatedAnimsOffset(_fbh) + calcInterpolatedAnimsByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV0::calcBoneNamesOffset(const scene::CFinalBoneHierarchy* _fbh)
{
	return calcNonInterpolatedAnimsOffset(_fbh) + calcNonInterpolatedAnimsByteSize(_fbh);
}

size_t FinalBoneHierarchyBlobV0::calcBonesByteSize(const scene::CFinalBoneHierarchy * _fbh)
{
	return _fbh->getBoneCount()*sizeof(*_fbh->getBoneData());
}
size_t FinalBoneHierarchyBlobV0::calcLevelsByteSize(const scene::CFinalBoneHierarchy * _fbh)
{
	return _fbh->getHierarchyLevels()*sizeof(*_fbh->getBoneTreeLevelEnd());
}
size_t FinalBoneHierarchyBlobV0::calcKeyFramesByteSize(const scene::CFinalBoneHierarchy * _fbh)
{
	return _fbh->getKeyFrameCount()*sizeof(*_fbh->getKeys());
}
size_t FinalBoneHierarchyBlobV0::calcInterpolatedAnimsByteSize(const scene::CFinalBoneHierarchy * _fbh)
{
	return _fbh->getAnimationCount()*sizeof(*_fbh->getInterpolatedAnimationData());
}
size_t FinalBoneHierarchyBlobV0::calcNonInterpolatedAnimsByteSize(const scene::CFinalBoneHierarchy * _fbh)
{
	return _fbh->getAnimationCount()*sizeof(*_fbh->getNonInterpolatedAnimationData());
}
size_t FinalBoneHierarchyBlobV0::calcBoneNamesByteSize(const scene::CFinalBoneHierarchy * _fbh)
{
	return _fbh->getSizeOfAllBoneNames();
}

size_t FinalBoneHierarchyBlobV0::calcBonesOffset() const
{
	return sizeof(FinalBoneHierarchyBlobV0);
}
size_t FinalBoneHierarchyBlobV0::calcLevelsOffset() const
{
	return calcBonesOffset() + calcBonesByteSize();
}
size_t FinalBoneHierarchyBlobV0::calcKeyFramesOffset() const
{
	return calcLevelsOffset() + calcLevelsByteSize();
}
size_t FinalBoneHierarchyBlobV0::calcInterpolatedAnimsOffset() const
{
	return calcKeyFramesOffset() + calcKeyFramesByteSize();
}
size_t FinalBoneHierarchyBlobV0::calcNonInterpolatedAnimsOffset() const
{
	return calcInterpolatedAnimsOffset() + calcInterpolatedAnimsByteSize();
}
size_t FinalBoneHierarchyBlobV0::calcBoneNamesOffset() const
{
	return calcNonInterpolatedAnimsOffset() + calcNonInterpolatedAnimsByteSize();
}

size_t FinalBoneHierarchyBlobV0::calcBonesByteSize() const
{
	return boneCount * scene::CFinalBoneHierarchy::getSizeOfSingleBone();
}
size_t FinalBoneHierarchyBlobV0::calcLevelsByteSize() const
{
	return numLevelsInHierarchy * sizeof(size_t);
}
size_t FinalBoneHierarchyBlobV0::calcKeyFramesByteSize() const
{
	return keyframeCount * sizeof(float);
}
size_t FinalBoneHierarchyBlobV0::calcInterpolatedAnimsByteSize() const
{
	return keyframeCount * boneCount * scene::CFinalBoneHierarchy::getSizeOfSingleAnimationData();
}
size_t FinalBoneHierarchyBlobV0::calcNonInterpolatedAnimsByteSize() const
{
	return keyframeCount * boneCount * scene::CFinalBoneHierarchy::getSizeOfSingleAnimationData();
}

}} // irr::core
