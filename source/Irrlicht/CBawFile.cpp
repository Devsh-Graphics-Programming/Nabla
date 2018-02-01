// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBawFile.h"

#include "ISkinnedMesh.h"
#include "SSkinMeshBuffer.h"
#include "CFinalBoneHierarchy.h"

//! for C++11
//using namespace std;


namespace irr { namespace core
{

	MeshBlobV1::MeshBlobV1(const aabbox3df& _box, uint32_t _cnt) : box(_box), meshBufCnt(_cnt)
	{
	}

	SkinnedMeshBlobV1::SkinnedMeshBlobV1(scene::CFinalBoneHierarchy* _fbh, const aabbox3df & _box, uint32_t _cnt)
		: boneHierarchyPtr(reinterpret_cast<uint64_t>(_fbh)), box(_box), meshBufCnt(_cnt)
	{
	}

	MeshBufferBlobV1::MeshBufferBlobV1(const scene::ICPUMeshBuffer* _mb)
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

	SkinnedMeshBufferBlobV1::SkinnedMeshBufferBlobV1(const scene::SCPUSkinMeshBuffer* _smb) : MeshBufferBlobV1(_smb)
	{
		indexValMin = _smb->getIndexMinBound();
		indexValMax = _smb->getIndexMaxBound();
		maxVertexBoneInfluences = _smb->getMaxVertexBoneInfluences();
	}

	MeshDataFormatDescBlobV1::MeshDataFormatDescBlobV1(const scene::IMeshDataFormatDesc<core::ICPUBuffer>* _desc)
	{
		using namespace scene;

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

	FinalBoneHierarchyBlobV1::FinalBoneHierarchyBlobV1(size_t _bCnt, size_t _numLvls, size_t _kfCnt) 
		: boneCount(_bCnt), numLevelsInHierarchy(_numLvls), keyframeCount(_kfCnt)
	{}

}} // irr::core
