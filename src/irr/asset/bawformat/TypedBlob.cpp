// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "irr/asset/bawformat/CBAWFile.h"

#include "IFileSystem.h"
#include "irr/video/CGPUMesh.h"
#include "irr/asset/CCPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"
#include "irr/asset/CCPUSkinnedMesh.h"
#include "irr/asset/bawformat/CBlobsLoadingManager.h"
#include "irr/asset/ICPUTexture.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "irr/asset/CBAWMeshFileLoader.h"

// The file consists of functions required by
// the latest blob version. 

namespace irr
{
namespace asset
{


namespace impl
{

template<class T>
inline core::smart_refctd_ptr<T> castPtrAndRefcount(void* ptr)
{
	return core::smart_refctd_ptr<T>(reinterpret_cast<T*>(ptr));
}

}


// Loading-related blobs' function implementations
template<>
core::unordered_set<uint64_t> TypedBlob<RawBufferBlobV3, asset::ICPUBuffer>::getNeededDeps(const void* _blob)
{
	return core::unordered_set<uint64_t>();
}

template<>
void* TypedBlob<RawBufferBlobV3, asset::ICPUBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	auto* blob = (RawBufferBlobV3*)_blob;
	asset::ICPUBuffer* buf = new asset::ICPUBuffer(_blobSize);
	memcpy(buf->getPointer(), blob->getData(), _blobSize);

	return buf;
}

template<>
void* TypedBlob<RawBufferBlobV3, asset::ICPUBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<RawBufferBlobV3, asset::ICPUBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUBuffer*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<TexturePathBlobV3, asset::ICPUTexture>::getNeededDeps(const void* _blob)
{
	return core::unordered_set<uint64_t>();
}

template<>
void* TypedBlob<TexturePathBlobV3, asset::ICPUTexture>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob || !_params.fs || !_params.ldr || !_params.manager)
		return nullptr;

	auto* blob = (TexturePathBlobV3*)_blob;

    // set ECF_DONT_CACHE_TOP_LEVEL flag because it will get cached in BAW loader
    asset::IAssetLoader::SAssetLoadParams params(_params.params.decryptionKeyLen, _params.params.decryptionKey, asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL, _params.filePath.c_str());
	constexpr uint32_t hierarchyLevel = 0u; // due to the above comment, absolutely meaningless right now

	const char* const texname = (const char*)blob->getData();
	auto bundle = static_cast<CBAWMeshFileLoader*>(_params.ldr)->interm_getAssetInHierarchy(_params.manager, std::string(texname), params, hierarchyLevel, _params.loaderOverride);

	auto assetRange = bundle.getContents();
	if (assetRange.first != assetRange.second)
	{
		auto texture = assetRange.first->get();
		texture->grab();
		return texture;
	}
	else
		return nullptr;
}

template<>
void* TypedBlob<TexturePathBlobV3, asset::ICPUTexture>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<TexturePathBlobV3, asset::ICPUTexture>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUTexture*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<MeshBlobV3, asset::ICPUMesh>::getNeededDeps(const void* _blob)
{
	auto* blob = (MeshBlobV3*)_blob;
	core::unordered_set<uint64_t> deps;
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
			deps.insert(blob->meshBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<MeshBlobV3, asset::ICPUMesh>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const auto* blob = (const MeshBlobV3*)_blob;
	asset::CCPUMesh* mesh = new asset::CCPUMesh();
	mesh->setBoundingBox(blob->box);

	return mesh;
}

template<>
void* TypedBlob<MeshBlobV3, asset::ICPUMesh>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const auto* blob = reinterpret_cast<const MeshBlobV3*>(_blob);
	asset::CCPUMesh* mesh = (asset::CCPUMesh*)_obj;
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(impl::castPtrAndRefcount<asset::ICPUMeshBuffer>(_deps[blob->meshBufPtrs[i]]));

	bool isRightHandedCoordinateSystem = blob->meshFlags & MeshBlobV3::EBMF_RIGHT_HANDED;
	if (isRightHandedCoordinateSystem != (bool)(_params.params.loaderFlags & IAssetLoader::E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES))
		_params.meshesToFlip.push(core::smart_refctd_ptr<ICPUMesh>(mesh));

	return _obj;
}

template<>
void TypedBlob<MeshBlobV3, asset::ICPUMesh>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUMesh*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<SkinnedMeshBlobV3, asset::ICPUSkinnedMesh>::getNeededDeps(const void* _blob)
{
	auto* blob = (SkinnedMeshBlobV3*)_blob;
	core::unordered_set<uint64_t> deps;
	deps.insert(blob->boneHierarchyPtr);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		if (blob->meshBufPtrs[i])
			deps.insert(blob->meshBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<SkinnedMeshBlobV3, asset::ICPUSkinnedMesh>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return NULL;

	const auto* blob = (const SkinnedMeshBlobV3*)_blob;
	asset::CCPUSkinnedMesh* mesh = new asset::CCPUSkinnedMesh();
	mesh->setBoundingBox(blob->box);

	return mesh;
}

template<>
void* TypedBlob<SkinnedMeshBlobV3, asset::ICPUSkinnedMesh>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return NULL;

	const auto* blob = (const SkinnedMeshBlobV3*)_blob;
	asset::CCPUSkinnedMesh* mesh = reinterpret_cast<asset::CCPUSkinnedMesh*>(_obj);
	mesh->setBoneReferenceHierarchy(impl::castPtrAndRefcount<CFinalBoneHierarchy>(_deps[blob->boneHierarchyPtr]));
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(impl::castPtrAndRefcount<asset::ICPUSkinnedMeshBuffer>(_deps[blob->meshBufPtrs[i]]));

	bool isRightHandedCoordinateSystem = blob->meshFlags & SkinnedMeshBlobV3::EBMF_RIGHT_HANDED;
	if (isRightHandedCoordinateSystem != (bool)(_params.params.loaderFlags & IAssetLoader::E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES) && mesh->getMeshType() == asset::EMT_ANIMATED_SKINNED)
		_params.meshesToFlip.push(core::smart_refctd_ptr<ICPUMesh>(mesh));

	return _obj;
}

template<>
void TypedBlob<SkinnedMeshBlobV3, asset::ICPUSkinnedMesh>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUSkinnedMesh*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::getNeededDeps(const void* _blob)
{
	auto* blob = (MeshBufferBlobV3*)_blob;
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
void* TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return nullptr;

	const auto* blob = (const MeshBufferBlobV3*)_blob;
	asset::ICPUMeshBuffer* buf = new asset::ICPUMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SCPUMaterial));
	buf->getMaterial().setBitfields(*(blob)->mat.bitfieldsPtr());
	for (size_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		memset(&buf->getMaterial().TextureLayer[i].Texture,0,sizeof(const void*));
		buf->getMaterial().TextureLayer[i].SamplingParams.setBitfields(*(blob)->mat.TextureLayer[i].SamplingParams.bitfieldsPtr());
	}

	buf->setBoundingBox(blob->box);
	buf->setIndexType((asset::E_INDEX_TYPE)blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType((asset::E_PRIMITIVE_TYPE)blob->primitiveType);
	buf->setPositionAttributeIx((asset::E_VERTEX_ATTRIBUTE_ID)blob->posAttrId);
	buf->setNormalnAttributeIx((asset::E_VERTEX_ATTRIBUTE_ID)blob->normalAttrId);

	return buf;
}

template<>
void* TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return nullptr;

	const auto* blob = (const MeshBufferBlobV3*)_blob;
	asset::ICPUMeshBuffer* buf = reinterpret_cast<asset::ICPUMeshBuffer*>(_obj);
	buf->setMeshDataAndFormat(impl::castPtrAndRefcount<asset::IMeshDataFormatDesc<asset::ICPUBuffer> >(_deps[blob->descPtr]));
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(blob->mat.getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, impl::castPtrAndRefcount<asset::ICPUTexture>(_deps[tex]));
	}

	return _obj;
}

template<>
void TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUMeshBuffer*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<SkinnedMeshBufferBlobV3, asset::ICPUSkinnedMeshBuffer>::getNeededDeps(const void* _blob)
{
	return TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::getNeededDeps(_blob);
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV3, asset::ICPUSkinnedMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return nullptr;

	const auto* blob = (const SkinnedMeshBufferBlobV3*)_blob;
	asset::ICPUSkinnedMeshBuffer* buf = new asset::ICPUSkinnedMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SCPUMaterial));
	buf->getMaterial().setBitfields(*(blob)->mat.bitfieldsPtr());
	for (size_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		memset(&buf->getMaterial().TextureLayer[i].Texture, 0, sizeof(const void*));
		buf->getMaterial().TextureLayer[i].SamplingParams.setBitfields(*(blob)->mat.TextureLayer[i].SamplingParams.bitfieldsPtr());
	}

	buf->setBoundingBox(blob->box);
	buf->setIndexType((asset::E_INDEX_TYPE)blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType((asset::E_PRIMITIVE_TYPE)blob->primitiveType);
	buf->setPositionAttributeIx((asset::E_VERTEX_ATTRIBUTE_ID)blob->posAttrId);
	buf->setNormalnAttributeIx((asset::E_VERTEX_ATTRIBUTE_ID)blob->normalAttrId);
	buf->setIndexRange(blob->indexValMin, blob->indexValMax);
	buf->setMaxVertexBoneInfluences(blob->maxVertexBoneInfluences);

	return buf;
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV3, asset::ICPUSkinnedMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return nullptr;

	const auto* blob = (const SkinnedMeshBufferBlobV3*)_blob;
	asset::ICPUSkinnedMeshBuffer* buf = reinterpret_cast<asset::ICPUSkinnedMeshBuffer*>(_obj);
	buf->setMeshDataAndFormat(impl::castPtrAndRefcount<asset::IMeshDataFormatDesc<asset::ICPUBuffer> >(_deps[blob->descPtr]));
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(blob->mat.getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, impl::castPtrAndRefcount<asset::ICPUTexture>(_deps[tex]));
	}

	return _obj;
}

template<>
void TypedBlob<SkinnedMeshBufferBlobV3, asset::ICPUSkinnedMeshBuffer>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::ICPUSkinnedMeshBuffer*>(_obj)->drop();
}

template<>
core::unordered_set<uint64_t> TypedBlob<FinalBoneHierarchyBlobV3, CFinalBoneHierarchy>::getNeededDeps(const void* _blob)
{
	return core::unordered_set<uint64_t>();
}

template<>
void* TypedBlob<FinalBoneHierarchyBlobV3, CFinalBoneHierarchy>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return nullptr;

	const uint8_t* const data = (const uint8_t*)_blob;
	const auto* blob = (const FinalBoneHierarchyBlobV3*)_blob;

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

	CFinalBoneHierarchy* fbh = new CFinalBoneHierarchy(
		bonesBegin, bonesEnd,
		boneNames, boneNames + blob->boneCount,
		(const size_t*)levelsBegin, (const size_t*)levelsEnd,
		(const float*)keyframesBegin, (const float*)keyframesEnd,
		interpolatedAnimsBegin, interpolatedAnimsEnd,
		nonInterpolatedAnimsBegin, nonInterpolatedAnimsEnd, blob->finalBoneHierarchyFlags&FinalBoneHierarchyBlobV3::EBFBHF_RIGHT_HANDED
	);

	if ((uint8_t*)boneNames == stack)
		for (size_t i = 0; i < blob->boneCount; ++i)
			boneNames[i].~string();
	else
		delete[] boneNames;

	return fbh;
}

template<>
void* TypedBlob<FinalBoneHierarchyBlobV3, CFinalBoneHierarchy>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	return _obj;
}

template<>
void TypedBlob<FinalBoneHierarchyBlobV3, CFinalBoneHierarchy>::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const CFinalBoneHierarchy*>(_obj)->drop();
}



template<>
core::unordered_set<uint64_t> TypedBlob<MeshDataFormatDescBlobV3, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::getNeededDeps(const void* _blob)
{
	auto blob = (MeshDataFormatDescBlobV3*)_blob;
	core::unordered_set<uint64_t> deps;
	if (blob->idxBufPtr)
		deps.insert(blob->idxBufPtr);
	for (uint32_t i = 0; i < asset::EVAI_COUNT; ++i)
		if (blob->attrBufPtrs[i])
			deps.insert(blob->attrBufPtrs[i]);
	return deps;
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV3, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	return new asset::ICPUMeshDataFormatDesc();
}

template<>
void* TypedBlob<MeshDataFormatDescBlobV3, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return nullptr;

	const auto* blob = (const MeshDataFormatDescBlobV3*)_blob;
	asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = reinterpret_cast<asset::ICPUMeshDataFormatDesc*>(_obj);
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
	{
		if (blob->attrBufPtrs[(int)i])
			desc->setVertexAttrBuffer(
				impl::castPtrAndRefcount<asset::ICPUBuffer>(_deps[blob->attrBufPtrs[i]]),
				i,
				static_cast<asset::E_FORMAT>(blob->attrFormat[i]),
				blob->attrStride[i],
				blob->attrOffset[i],
				(blob->attrDivisor>>i)&1u
			);
	}
	if (blob->idxBufPtr)
		desc->setIndexBuffer(impl::castPtrAndRefcount<asset::ICPUBuffer>(_deps[blob->idxBufPtr]));
	return _obj;
}

template<>
void TypedBlob<MeshDataFormatDescBlobV3, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::releaseObj(const void* _obj)
{
	if (_obj)
		reinterpret_cast<const asset::IMeshDataFormatDesc<asset::ICPUBuffer>*>(_obj)->drop();
}


}} // irr:core
