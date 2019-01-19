// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "irr/asset/bawformat/CBAWFile.h"

#include "ISceneManager.h"
#include "IFileSystem.h"
#include "IVideoDriver.h"
#include "irr/video/SGPUMesh.h"
#include "irr/asset/SCPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"
#include "irr/asset/CCPUSkinnedMesh.h"
#include "irr/asset/bawformat/CBlobsLoadingManager.h"
#include "irr/asset/ICPUTexture.h"
#include "IrrlichtDevice.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "irr/asset/CBAWMeshFileLoader.h"

namespace irr { namespace asset
{

// Loading-related blobs' function implementations
template<>
core::unordered_set<uint64_t> TypedBlob<RawBufferBlobV0, asset::ICPUBuffer>::getNeededDeps(const void* _blob)
{
	return core::unordered_set<uint64_t>();
}

template<>
void* TypedBlob<RawBufferBlobV0, asset::ICPUBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	RawBufferBlobV0* blob = (RawBufferBlobV0*)_blob;
	asset::ICPUBuffer* buf = new asset::ICPUBuffer(_blobSize);
	memcpy(buf->getPointer(), blob->getData(), _blobSize);

	return buf;
}

template<>
void* TypedBlob<RawBufferBlobV0, asset::ICPUBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<RawBufferBlobV0, asset::ICPUBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUBuffer*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<TexturePathBlobV0, asset::ICPUTexture>::getNeededDeps(const void* _blob)
{
	return core::unordered_set<uint64_t>();
}

template<>
void* TypedBlob<TexturePathBlobV0, asset::ICPUTexture>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob || !_params.fs || !_params.ldr || !_params.device)
		return nullptr;

	TexturePathBlobV0* blob = (TexturePathBlobV0*)_blob;

    // set ECF_DONT_CACHE_TOP_LEVEL flag because it will get cached in BAW loader
    asset::IAssetLoader::SAssetLoadParams params(_params.params.decryptionKeyLen, _params.params.decryptionKey, asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

	asset::ICPUTexture* texture;
	const char* const texname = (const char*)blob->getData();
	if (_params.fs->existFile(texname))
	{
		texture = static_cast<asset::ICPUTexture*>(static_cast<CBAWMeshFileLoader*>(_params.ldr)->interm_getAssetInHierarchy(_params.device->getAssetManager(), texname, params, 0u, _params.loaderOverride));
	}
	else
	{
		const io::path path = _params.filePath + texname;
		// try to read from the path relative to where the .baw is loaded from
		texture = static_cast<asset::ICPUTexture*>(static_cast<CBAWMeshFileLoader*>(_params.ldr)->interm_getAssetInHierarchy(_params.device->getAssetManager(), path.c_str(), params, 0u, _params.loaderOverride));
	}

	return texture;
}

template<>
void* TypedBlob<TexturePathBlobV0, asset::ICPUTexture>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<TexturePathBlobV0, asset::ICPUTexture>::releaseObj(const void* _obj)
{
}

template<>
core::unordered_set<uint64_t> TypedBlob<MeshBlobV0, asset::ICPUMesh>::getNeededDeps(const void* _blob)
{
	MeshBlobV0* blob = (MeshBlobV0*)_blob;
	core::unordered_set<uint64_t> deps;
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
			deps.insert(blob->meshBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<MeshBlobV0, asset::ICPUMesh>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const MeshBlobV0* blob = (const MeshBlobV0*)_blob;
	asset::SCPUMesh* mesh = new asset::SCPUMesh();
	mesh->setBoundingBox(blob->box);

	return mesh;
}

template<>
void* TypedBlob<MeshBlobV0, asset::ICPUMesh>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const MeshBlobV0* blob = reinterpret_cast<const MeshBlobV0*>(_blob);
	asset::SCPUMesh* mesh = (asset::SCPUMesh*)_obj;
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(reinterpret_cast<asset::ICPUMeshBuffer*>(_deps[blob->meshBufPtrs[i]]));
	return _obj;
}

template<>
void TypedBlob<MeshBlobV0, asset::ICPUMesh>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUMesh*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<SkinnedMeshBlobV0, asset::ICPUSkinnedMesh>::getNeededDeps(const void* _blob)
{
	SkinnedMeshBlobV0* blob = (SkinnedMeshBlobV0*)_blob;
	core::unordered_set<uint64_t> deps;
	deps.insert(blob->boneHierarchyPtr);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
			deps.insert(blob->meshBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<SkinnedMeshBlobV0, asset::ICPUSkinnedMesh>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const SkinnedMeshBlobV0* blob = (const SkinnedMeshBlobV0*)_blob;
	asset::CCPUSkinnedMesh* mesh = new asset::CCPUSkinnedMesh();
	mesh->setBoundingBox(blob->box);

	return mesh;
}

template<>
void* TypedBlob<SkinnedMeshBlobV0, asset::ICPUSkinnedMesh>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const SkinnedMeshBlobV0* blob = (const SkinnedMeshBlobV0*)_blob;
	asset::CCPUSkinnedMesh* mesh = reinterpret_cast<asset::CCPUSkinnedMesh*>(_obj);
	mesh->setBoneReferenceHierarchy(reinterpret_cast<scene::CFinalBoneHierarchy*>(_deps[blob->boneHierarchyPtr]));
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(reinterpret_cast<asset::ICPUSkinnedMeshBuffer*>(_deps[blob->meshBufPtrs[i]]));

	return _obj;
}

template<>
void TypedBlob<SkinnedMeshBlobV0, asset::ICPUSkinnedMesh>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUSkinnedMesh*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>::getNeededDeps(const void* _blob)
{
	MeshBufferBlobV0* blob = (MeshBufferBlobV0*)_blob;
	core::unordered_set<uint64_t> deps;
	deps.insert(blob->descPtr);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(blob->mat.getTexture(i));
		if (tex)
			deps.insert(tex);
	}
	return deps;
}

template<>
void* TypedBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const MeshBufferBlobV0* blob = (const MeshBufferBlobV0*)_blob;
	asset::ICPUMeshBuffer* buf = new asset::ICPUMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SGPUMaterial));
	buf->getMaterial().setBitfields(*(blob)->mat.bitfieldsPtr());
	for (size_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
		buf->getMaterial().TextureLayer[i].SamplingParams.setBitfields(*(blob)->mat.TextureLayer[i].SamplingParams.bitfieldsPtr());

	buf->setBoundingBox(blob->box);
	buf->setIndexType((asset::E_INDEX_TYPE)blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType((asset::E_PRIMITIVE_TYPE)blob->primitiveType);
	buf->setPositionAttributeIx((asset::E_VERTEX_ATTRIBUTE_ID)blob->posAttrId);

	return buf;
}

template<>
void* TypedBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const MeshBufferBlobV0* blob = (const MeshBufferBlobV0*)_blob;
	asset::ICPUMeshBuffer* buf = reinterpret_cast<asset::ICPUMeshBuffer*>(_obj);
	buf->setMeshDataAndFormat(reinterpret_cast<asset::IMeshDataFormatDesc<asset::ICPUBuffer>*>(_deps[blob->descPtr]));
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, reinterpret_cast<asset::ICPUTexture*>(_deps[tex]));
	}
	return _obj;
}

template<>
void TypedBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUMeshBuffer*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<SkinnedMeshBufferBlobV0, asset::ICPUSkinnedMeshBuffer>::getNeededDeps(const void* _blob)
{
	return TypedBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>::getNeededDeps(_blob);
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV0, asset::ICPUSkinnedMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const SkinnedMeshBufferBlobV0* blob = (const SkinnedMeshBufferBlobV0*)_blob;
	asset::ICPUSkinnedMeshBuffer* buf = new asset::ICPUSkinnedMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SGPUMaterial));
	buf->getMaterial().setBitfields(*(blob)->mat.bitfieldsPtr());
	for (size_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
		buf->getMaterial().TextureLayer[i].SamplingParams.setBitfields(*(blob)->mat.TextureLayer[i].SamplingParams.bitfieldsPtr());

	buf->setBoundingBox(blob->box);
	buf->setIndexType((asset::E_INDEX_TYPE)blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType((asset::E_PRIMITIVE_TYPE)blob->primitiveType);
	buf->setPositionAttributeIx((asset::E_VERTEX_ATTRIBUTE_ID)blob->posAttrId);
	buf->setIndexRange(blob->indexValMin, blob->indexValMax);
	buf->setMaxVertexBoneInfluences(blob->maxVertexBoneInfluences);

	return buf;
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV0, asset::ICPUSkinnedMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const SkinnedMeshBufferBlobV0* blob = (const SkinnedMeshBufferBlobV0*)_blob;
	asset::ICPUSkinnedMeshBuffer* buf = reinterpret_cast<asset::ICPUSkinnedMeshBuffer*>(_obj);
	buf->setMeshDataAndFormat(reinterpret_cast<asset::IMeshDataFormatDesc<asset::ICPUBuffer>*>(_deps[blob->descPtr]));
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
		{
			buf->getMaterial().setTexture(i, reinterpret_cast<asset::ICPUTexture*>(_deps[tex]));
		}
	}
	return _obj;
}

template<>
void TypedBlob<SkinnedMeshBufferBlobV0, asset::ICPUSkinnedMeshBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUSkinnedMeshBuffer*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::getNeededDeps(const void* _blob)
{
	return core::unordered_set<uint64_t>();
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
	core::stringc* boneNames = (blob->boneCount * sizeof(core::stringc) <= sizeof(stack)) ? (core::stringc*)stack : new core::stringc[blob->boneCount];

	for (size_t i = 0; i < blob->boneCount; ++i)
	{
		size_t len = strlen(strPtr) + 1;
		_IRR_DEBUG_BREAK_IF(strPtr + len > blobEnd)
		if ((uint8_t*)boneNames == stack)
			new (boneNames + i) core::stringc(strPtr);
		else
			boneNames[i] = core::stringc(strPtr);
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
void* TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const scene::CFinalBoneHierarchy*>(_obj)->drop();
}



template<>
core::unordered_set<uint64_t> TypedBlob<MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::getNeededDeps(const void* _blob)
{
	MeshDataFormatDescBlobV1* blob = (MeshDataFormatDescBlobV1*)_blob;
	core::unordered_set<uint64_t> deps;
	if (blob->idxBufPtr)
		deps.insert(blob->idxBufPtr);
	for (uint32_t i = 0; i < asset::EVAI_COUNT; ++i)
		if (blob->attrBufPtrs[i])
			deps.insert(blob->attrBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	return new asset::ICPUMeshDataFormatDesc();
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	using namespace scene;

	if (!_obj || !_blob)
		return NULL;

	const MeshDataFormatDescBlobV1* blob = (const MeshDataFormatDescBlobV1*)_blob;
	asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = reinterpret_cast<asset::ICPUMeshDataFormatDesc*>(_obj);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
	{
		if (blob->attrBufPtrs[(int)i])
			desc->setVertexAttrBuffer(
				reinterpret_cast<asset::ICPUBuffer*>(_deps[blob->attrBufPtrs[i]]),
				i,
				static_cast<asset::E_FORMAT>(blob->attrFormat[i]),
				blob->attrStride[i],
				blob->attrOffset[i],
				(blob->attrDivisor>>i)&1u
			);
	}
	if (blob->idxBufPtr)
		desc->setIndexBuffer(reinterpret_cast<asset::ICPUBuffer*>(_deps[blob->idxBufPtr]));
	return _obj;
}

template<>
void TypedBlob<MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::IMeshDataFormatDesc<asset::ICPUBuffer>*>(_obj)->drop();
}


}} // irr:core
