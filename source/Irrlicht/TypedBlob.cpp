// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWFile.h"

#include "ISceneManager.h"
#include "IFileSystem.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"
#include "CBlobsLoadingManager.h"

namespace irr { namespace core
{

// Loading-related blobs' function implementations
template<>
std::vector<uint64_t> TypedBlob<RawBufferBlobV0, ICPUBuffer>::getNeededDeps(const void* _blob)
{
	return std::vector<uint64_t>();
}

template<>
void* TypedBlob<RawBufferBlobV0, ICPUBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	RawBufferBlobV0* blob = (RawBufferBlobV0*)_blob;
	core::ICPUBuffer* buf = new core::ICPUBuffer(_blobSize);
	memcpy(buf->getPointer(), blob->getData(), _blobSize);

	return buf;
}

template<>
void* TypedBlob<RawBufferBlobV0, ICPUBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<RawBufferBlobV0, ICPUBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const ICPUBuffer*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<TexturePathBlobV0, video::IVirtualTexture>::getNeededDeps(const void* _blob)
{
	return std::vector<uint64_t>();
}

template<>
void* TypedBlob<TexturePathBlobV0, video::IVirtualTexture>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob || !_params.fs || !_params.sm)
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
void* TypedBlob<TexturePathBlobV0, video::IVirtualTexture>::finalize(void* _obj, const void* _blob, size_t _blobSize,std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<TexturePathBlobV0, video::IVirtualTexture>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const video::IVirtualTexture*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<MeshBlobV0, scene::ICPUMesh>::getNeededDeps(const void* _blob)
{
	MeshBlobV0* blob = (MeshBlobV0*)_blob;
	std::vector<uint64_t> deps;
	deps.reserve(blob->meshBufCnt);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
			deps.push_back(blob->meshBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<MeshBlobV0, scene::ICPUMesh>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const MeshBlobV0* blob = (const MeshBlobV0*)_blob;
	scene::SCPUMesh* mesh = new scene::SCPUMesh();
	mesh->setBoundingBox(blob->box);

	return mesh;
}

template<>
void* TypedBlob<MeshBlobV0, scene::ICPUMesh>::finalize(void* _obj, const void* _blob, size_t _blobSize, std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const MeshBlobV0* blob = reinterpret_cast<const MeshBlobV0*>(_blob);
	scene::SCPUMesh* mesh = (scene::SCPUMesh*)_obj;
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(reinterpret_cast<scene::ICPUMeshBuffer*>(_deps[blob->meshBufPtrs[i]]));
	return _obj;
}

template<>
void TypedBlob<MeshBlobV0, scene::ICPUMesh>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::ICPUMesh*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::getNeededDeps(const void* _blob)
{
	SkinnedMeshBlobV0* blob = (SkinnedMeshBlobV0*)_blob;
	std::vector<uint64_t> deps;
	deps.reserve(blob->meshBufCnt + 1);
	deps.push_back(blob->boneHierarchyPtr);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
			deps.push_back(blob->meshBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const SkinnedMeshBlobV0* blob = (const SkinnedMeshBlobV0*)_blob;
	scene::CCPUSkinnedMesh* mesh = new scene::CCPUSkinnedMesh();
	mesh->setBoundingBox(blob->box);

	return mesh;
}

template<>
void* TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::finalize(void* _obj, const void* _blob, size_t _blobSize,std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const SkinnedMeshBlobV0* blob = (const SkinnedMeshBlobV0*)_blob;
	scene::CCPUSkinnedMesh* mesh = reinterpret_cast<scene::CCPUSkinnedMesh*>(_obj);
	mesh->setBoneReferenceHierarchy(reinterpret_cast<scene::CFinalBoneHierarchy*>(_deps[blob->boneHierarchyPtr]));
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(reinterpret_cast<scene::SCPUSkinMeshBuffer*>(_deps[blob->meshBufPtrs[i]]));

	return _obj;
}

template<>
void TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::ICPUSkinnedMesh*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::getNeededDeps(const void* _blob)
{
	MeshBufferBlobV0* blob = (MeshBufferBlobV0*)_blob;
	std::vector<uint64_t> deps;
	deps.reserve(_IRR_MATERIAL_MAX_TEXTURES_ + 1);
	deps.push_back(blob->descPtr);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(blob->mat.getTexture(i));
		if (tex)
			deps.push_back(tex);
	}
	return deps;
}

template<>
void* TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const MeshBufferBlobV0* blob = (const MeshBufferBlobV0*)_blob;
	scene::ICPUMeshBuffer* buf = new scene::ICPUMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setIndexType((video::E_INDEX_TYPE)blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType((scene::E_PRIMITIVE_TYPE)blob->primitiveType);
	buf->setPositionAttributeIx((scene::E_VERTEX_ATTRIBUTE_ID)blob->posAttrId);

	return buf;
}

template<>
void* TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize, std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const MeshBufferBlobV0* blob = (const MeshBufferBlobV0*)_blob;
	scene::ICPUMeshBuffer* buf = reinterpret_cast<scene::ICPUMeshBuffer*>(_obj);
	buf->setMeshDataAndFormat(reinterpret_cast<scene::IMeshDataFormatDesc<core::ICPUBuffer>*>(_deps[blob->descPtr]));
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, reinterpret_cast<video::ITexture*>(_deps[tex])); // ITexture* since VideoDriver returns ITexture*
	}
	return _obj;
}

template<>
void TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::ICPUMeshBuffer*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::getNeededDeps(const void* _blob)
{
	return TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>::getNeededDeps(_blob);
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const SkinnedMeshBufferBlobV0* blob = (const SkinnedMeshBufferBlobV0*)_blob;
	scene::SCPUSkinMeshBuffer* buf = new scene::SCPUSkinMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setIndexType((video::E_INDEX_TYPE)blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType((scene::E_PRIMITIVE_TYPE)blob->primitiveType);
	buf->setPositionAttributeIx((scene::E_VERTEX_ATTRIBUTE_ID)blob->posAttrId);
	buf->setIndexRange(blob->indexValMin, blob->indexValMax);
	buf->setMaxVertexBoneInfluences(blob->maxVertexBoneInfluences);

	return buf;
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const SkinnedMeshBufferBlobV0* blob = (const SkinnedMeshBufferBlobV0*)_blob;
	scene::SCPUSkinMeshBuffer* buf = reinterpret_cast<scene::SCPUSkinMeshBuffer*>(_obj);
	buf->setMeshDataAndFormat(reinterpret_cast<scene::IMeshDataFormatDesc<core::ICPUBuffer>*>(_deps[blob->descPtr]));
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
		{
			buf->getMaterial().setTexture(i, reinterpret_cast<video::ITexture*>(_deps[tex])); // ITexture* since VideoDriver returns ITexture*
		}
	}
	return _obj;
}

template<>
void TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::SCPUSkinMeshBuffer*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::getNeededDeps(const void* _blob)
{
	MeshDataFormatDescBlobV0* blob = (MeshDataFormatDescBlobV0*)_blob;
	std::vector<uint64_t> deps;
	deps.reserve(scene::EVAI_COUNT + 1);
	if (blob->idxBufPtr)
		deps.push_back(blob->idxBufPtr);
	for (uint32_t i = 0; i < scene::EVAI_COUNT; ++i)
		if (blob->attrBufPtrs[i])
			deps.push_back(blob->attrBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	return new scene::ICPUMeshDataFormatDesc();
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::finalize(void* _obj, const void* _blob, size_t _blobSize, std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	using namespace scene;

	if (!_obj || !_blob)
		return NULL;

	const MeshDataFormatDescBlobV0* blob = (const MeshDataFormatDescBlobV0*)_blob;
	scene::IMeshDataFormatDesc<core::ICPUBuffer>* desc = reinterpret_cast<scene::ICPUMeshDataFormatDesc*>(_obj);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
	{
		if (blob->attrBufPtrs[(int)i])
			desc->mapVertexAttrBuffer(
				reinterpret_cast<ICPUBuffer*>(_deps[blob->attrBufPtrs[(int)i]]),
				i,
				(scene::E_COMPONENTS_PER_ATTRIBUTE)blob->cpa[(int)i],
				(scene::E_COMPONENT_TYPE)blob->attrType[(int)i],
				blob->attrStride[(int)i],
				blob->attrOffset[(int)i],
				blob->attrDivisor[(int)i]
			);
	}
	if (blob->idxBufPtr)
		desc->mapIndexBuffer(reinterpret_cast<ICPUBuffer*>(_deps[blob->idxBufPtr]));
	return _obj;
}

template<>
void TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::IMeshDataFormatDesc<core::ICPUBuffer>*>(_obj)->drop();
}

template<>
std::vector<uint64_t> TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::getNeededDeps(const void* _blob)
{
	return std::vector<uint64_t>();
}

template<>
void* TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const uint8_t* const data = (const uint8_t*)_blob;
	const FinalBoneHierarchyBlobV0* blob = (const FinalBoneHierarchyBlobV0*)_blob;

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

template<>
void* TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::finalize(void* _obj, const void* _blob, size_t _blobSize, std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::CFinalBoneHierarchy*>(_obj)->drop();
}


}} // irr:core
