// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "irr/asset/bawformat/CBAWFile.h"

#include "IFileSystem.h"
#include "irr/asset/asset.h"

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

	return deps;
}

template<>
void* TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
	if (!_blob)
		return nullptr;

	const auto* blob = (const MeshBufferBlobV3*)_blob;
	asset::ICPUMeshBuffer* buf = new asset::ICPUMeshBuffer();

	return buf;
}

template<>
void* TypedBlob<MeshBufferBlobV3, asset::ICPUMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return nullptr;

	const auto* blob = (const MeshBufferBlobV3*)_blob;
	asset::ICPUMeshBuffer* buf = reinterpret_cast<asset::ICPUMeshBuffer*>(_obj);

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

	return buf;
}

template<>
void* TypedBlob<SkinnedMeshBufferBlobV3, asset::ICPUSkinnedMeshBuffer>::finalize(void* _obj, const void* _blob, size_t _blobSize,core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
	if (!_obj || !_blob)
		return nullptr;

	const auto* blob = (const SkinnedMeshBufferBlobV3*)_blob;
	asset::ICPUSkinnedMeshBuffer* buf = reinterpret_cast<asset::ICPUSkinnedMeshBuffer*>(_obj);

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
		_NBL_DEBUG_BREAK_IF(strPtr + len > blobEnd)
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

}} // irr:core
