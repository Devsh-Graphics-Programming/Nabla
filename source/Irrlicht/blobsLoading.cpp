// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBawFile.h"

#include "ISceneManager.h"
#include "IFileSystem.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"

//! Adds support of given blob type to BlobsLoadingManager. For use ONLY inside BlobsLoadingManager's constructor.
#define _IRR_ADD_BLOB_SUPPORT_(ClassName, EnumValue, Version) \
m_evalFunctions[(uint32_t)Blob::EnumValue][Version] = &ClassName::getNeededDeps; \
m_makeFunctions[(uint32_t)Blob::EnumValue][Version] = &ClassName::tryMake;

namespace irr { namespace core
{

BlobsLoadingManager::BlobsLoadingManager(uint64_t _fileVer) : m_fileVer(_fileVer)
{
	_IRR_ADD_BLOB_SUPPORT_(RawBufferBlobV0, EBT_RAW_DATA_BUFFER, 0);
	_IRR_ADD_BLOB_SUPPORT_(TexturePathBlobV0, EBT_TEXTURE_PATH, 0);
	_IRR_ADD_BLOB_SUPPORT_(MeshBlobV0, EBT_MESH, 0);
	_IRR_ADD_BLOB_SUPPORT_(SkinnedMeshBlobV0, EBT_SKINNED_MESH, 0);
	_IRR_ADD_BLOB_SUPPORT_(MeshBufferBlobV0, EBT_MESH_BUFFER, 0);
	_IRR_ADD_BLOB_SUPPORT_(SkinnedMeshBufferBlobV0, EBT_SKINNED_MESH_BUFFER, 0);
	_IRR_ADD_BLOB_SUPPORT_(MeshDataFormatDescBlobV0, EBT_DATA_FORMAT_DESC, 0);
	_IRR_ADD_BLOB_SUPPORT_(FinalBoneHierarchyBlobV0, EBT_FINAL_BONE_HIERARCHY, 0);
}
#undef _IRR_ADD_BLOB_SUPPORT_

uint32_t BlobsLoadingManager::getNeededDeps(uint32_t _blobType, void * _blob, std::deque<uint64_t>& _q)
{
	return m_evalFunctions[_blobType][m_fileVer](_blob, _q);
}

void * BlobsLoadingManager::tryMake(uint32_t _blobType, void * _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams & _params)
{
#ifdef _DEBUG
	for (std::map<uint64_t, void*>::iterator it = _deps.begin(); it != _deps.end(); ++it)
	{
		_IRR_DEBUG_BREAK_IF(!it->second)
	}
#endif
	return m_makeFunctions[_blobType][m_fileVer](_blob, _blobSize, _deps, _params);
}

// Loading-related blobs' function implementations

template<>
uint32_t TypedBlob<RawBufferBlobV0, ICPUBuffer>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	return 0;
}

template<>
void* TypedBlob<RawBufferBlobV0, ICPUBuffer>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	RawBufferBlobV0* blob = (RawBufferBlobV0*)_blob;
	core::ICPUBuffer* buf = new core::ICPUBuffer(_blobSize);
	memcpy(buf->getPointer(), blob->getData(), _blobSize);

	return buf;
}

template<>
uint32_t TypedBlob<TexturePathBlobV0, video::IVirtualTexture>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	return 0;
}

template<>
void* TypedBlob<TexturePathBlobV0, video::IVirtualTexture>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	if (!_params.fs || !_params.sm)
		return NULL;

	TexturePathBlobV0* blob = (TexturePathBlobV0*)_blob;

	video::ITexture* texture;
	const char* const texname = (const char*)blob->getData();
	bool newTexture;
	if (_params.fs->existFile(texname))
	{
		newTexture = _params.sm->getVideoDriver()->findTexture(texname) == 0;
		texture = _params.sm->getVideoDriver()->getTexture(texname);
	}
	else
	{
		const io::path path = _params.filePath + texname;
		newTexture = _params.sm->getVideoDriver()->findTexture(path) == 0;
		// try to read from the path relative to where the .baw is loaded from
		texture = _params.sm->getVideoDriver()->getTexture(path);
	}
	//! @todo @bug Do somemthing with `newTexture`? In obj loader something happens in case where newTexture is true

	return texture;
}

template<>
uint32_t TypedBlob<MeshBlobV0, scene::ICPUMesh>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	MeshBlobV0* blob = (MeshBlobV0*)_blob;
	uint32_t needCnt = 0;
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
		{
			_q.push_back(blob->meshBufPtrs[i]);
			++needCnt;
		}
	return needCnt;
}

template<>
void* TypedBlob<MeshBlobV0, scene::ICPUMesh>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	MeshBlobV0* blob = (MeshBlobV0*)_blob;
	scene::SCPUMesh* mesh = new scene::SCPUMesh();
	mesh->setBoundingBox(blob->box);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(reinterpret_cast<scene::ICPUMeshBuffer*>(_deps[blob->meshBufPtrs[i]]));

	return mesh;
}

template<>
uint32_t TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	SkinnedMeshBlobV0* blob = (SkinnedMeshBlobV0*)_blob;
	uint32_t needCnt = 1;
	_q.push_back(blob->boneHierarchyPtr);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
		{
			_q.push_back(blob->meshBufPtrs[i]);
			++needCnt;
		}
	return needCnt;
}

template<>
void* TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	SkinnedMeshBlobV0* blob = (SkinnedMeshBlobV0*)_blob;
	scene::CCPUSkinnedMesh* mesh = new scene::CCPUSkinnedMesh();
	mesh->setBoneReferenceHierarchy(reinterpret_cast<scene::CFinalBoneHierarchy*>(_deps[blob->boneHierarchyPtr]));
	mesh->setBoundingBox(blob->box);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(reinterpret_cast<scene::SCPUSkinMeshBuffer*>(_deps[blob->meshBufPtrs[i]]));
	// shall i call mesh->finalize()?

	return mesh;
}

template<>
uint32_t TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	MeshBufferBlobV0* blob = (MeshBufferBlobV0*)_blob;
	uint32_t needCnt = 1;
	_q.push_back(blob->descPtr);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(blob->mat.getTexture(i));
		if (tex)
		{
			++needCnt;
			_q.push_back(tex);
		}
	}
	return needCnt;
}

template<>
void* TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	MeshBufferBlobV0* blob = (MeshBufferBlobV0*)_blob;
	scene::ICPUMeshBuffer* buf = new scene::ICPUMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setMeshDataAndFormat(reinterpret_cast<scene::IMeshDataFormatDesc<core::ICPUBuffer>*>(_deps[blob->descPtr]));
	buf->setIndexType(blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType(blob->primitiveType);
	buf->setPositionAttributeIx(blob->posAttrId);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, reinterpret_cast<video::IVirtualTexture*>(_deps[tex]));
	}

	return buf;
}

template<>
uint32_t TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	return TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::getNeededDeps(_blob, _q);
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	SkinnedMeshBufferBlobV0* blob = (SkinnedMeshBufferBlobV0*)_blob;
	scene::SCPUSkinMeshBuffer* buf = new scene::SCPUSkinMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setMeshDataAndFormat(reinterpret_cast<scene::IMeshDataFormatDesc<core::ICPUBuffer>*>(_deps[blob->descPtr]));
	buf->setIndexType(blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType(blob->primitiveType);
	buf->setPositionAttributeIx(blob->posAttrId);
	buf->setIndexRange(blob->indexValMin, blob->indexValMax);
	buf->setMaxVertexBoneInfluences(blob->maxVertexBoneInfluences);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, reinterpret_cast<video::IVirtualTexture*>(_deps[tex]));
	}

	return buf;
}

template<>
uint32_t TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	MeshDataFormatDescBlobV0* blob = (MeshDataFormatDescBlobV0*)_blob;
	uint32_t needCnt = blob->idxBufPtr ? 1 : 0;
	if (needCnt)
		_q.push_back(blob->idxBufPtr);
	for (uint32_t i = 0; i < scene::EVAI_COUNT; ++i)
		if (blob->attrBufPtrs[i])
		{
			++needCnt;
			_q.push_back(blob->attrBufPtrs[i]);
		}
	return needCnt;
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	using namespace scene;

	MeshDataFormatDescBlobV0* blob = (MeshDataFormatDescBlobV0*)_blob;
	scene::IMeshDataFormatDesc<core::ICPUBuffer>* desc = new scene::ICPUMeshDataFormatDesc();

	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
	{
		if (blob->attrBufPtrs[(int)i])
			desc->mapVertexAttrBuffer(
				reinterpret_cast<ICPUBuffer*>(_deps[blob->attrBufPtrs[(int)i]]),
				i,
				blob->cpa[(int)i],
				blob->attrType[(int)i],
				blob->attrStride[(int)i],
				blob->attrOffset[(int)i],
				blob->attrDivisor[(int)i]
			);
	}
	if (blob->idxBufPtr)
		desc->mapIndexBuffer(reinterpret_cast<ICPUBuffer*>(_deps[blob->idxBufPtr]));

	return desc;
}

template<>
uint32_t TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::getNeededDeps(void* _blob, std::deque<uint64_t>& _q)
{
	return 0;
}

template<>
void* TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params)
{
	const uint8_t* const data = (uint8_t*)_blob;
	FinalBoneHierarchyBlobV0* blob = (FinalBoneHierarchyBlobV0*)_blob;

	const uint8_t* const bonesBegin = data + blob->calcBonesOffset();
	const uint8_t* const bonesEnd = bonesBegin + blob->calcBonesByteSize();

	const uint8_t* const levelsBegin = data + blob->calcLevelsOffset();
	const uint8_t* const levelsEnd = levelsBegin + blob->calcLevelsByteSize();

	const uint8_t* const keyframesBegin = data + blob->calcKeyFramesOffset();
	const uint8_t* const keyframesEnd = keyframesBegin + blob->calcKeyFramesByteSize();

	const uint8_t* const interpolatedAnimsBegin = data + blob->calcInterpolatedAnimsOffset();
	const uint8_t* const interpolatedAnimsEnd = interpolatedAnimsBegin + blob->calcInterpolatedAnimsByteSize();

	const uint8_t* const nonInterpolatedAnimsBegin = data + blob->calcNonInterpolatedAnimsOffset();
	const uint8_t* const nonInterpolatedAnimsEnd = nonInterpolatedAnimsBegin + blob->calcNonInterpolatedAnimsByteSize();

	const uint8_t* const boneNamesBegin = data + blob->calcBoneNamesOffset();

	const char * strPtr = (const char*)boneNamesBegin;
	const char* const blobEnd = (const char*)(data + _blobSize);

	uint8_t stack[1u << 14];
	stringc* const boneNames = (blob->boneCount * sizeof(stringc) <= sizeof(stack)) ? (stringc*)stack : new stringc[blob->boneCount];
	for (size_t i = 0; i < blob->boneCount; ++i)
	{
		size_t len = strlen(strPtr) + 1;
		_IRR_DEBUG_BREAK_IF(strPtr + len > blobEnd)
		if ((uint8_t*)boneNames == stack)
			new (boneNames + i) stringc(strPtr);
		else
			boneNames[i] = stringc(strPtr);
		strPtr += len;
	}

	scene::CFinalBoneHierarchy* fbh = new scene::CFinalBoneHierarchy(
		bonesBegin, bonesEnd,
		boneNames, boneNames + blob->boneCount,
		(const size_t*)levelsBegin, (const size_t*)levelsEnd,
		(const float*)keyframesBegin, (const float*)keyframesEnd,
		interpolatedAnimsBegin, interpolatedAnimsEnd,
		nonInterpolatedAnimsBegin, nonInterpolatedAnimsEnd
	);

	if ((uint8_t*)boneNames == stack)
		for (size_t i = 0; i < blob->boneCount; ++i)
			boneNames[i].~string();
	else
		delete[] boneNames;

	return fbh;
}

}} // irr:core