// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CBAWMeshFileLoader.h"

#include <stack>

#include "os.h"
#include "CMemoryFile.h"
#include "CFinalBoneHierarchy.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "irr/asset/bawformat/legacy/CBAWVersionUpFunctions.h"
#include "irr/video/CGPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"

#include "lz4/lib/lz4.h"
#undef Bool
#include "lzma/C/LzmaDec.h"

namespace irr
{
namespace asset
{

struct LzmaMemMngmnt
{
        static void *alloc(ISzAllocPtr, size_t _size) { return _NBL_ALIGNED_MALLOC(_size,_NBL_SIMD_ALIGNMENT); }
        static void release(ISzAllocPtr, void* _addr) { _NBL_ALIGNED_FREE(_addr); }
    private:
        LzmaMemMngmnt() {}
};


CBAWMeshFileLoader::~CBAWMeshFileLoader()
{
}

CBAWMeshFileLoader::CBAWMeshFileLoader(IAssetManager* _manager) : m_manager(_manager), m_fileSystem(_manager->getFileSystem())
{
#ifdef _NBL_DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
}

SAssetBundle CBAWMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	return {};
}

bool CBAWMeshFileLoader::safeRead(io::IReadFile * _file, void * _buf, size_t _size) const
{
	if (_file->getPos() + _size > _file->getSize())
		return false;
	_file->read(_buf, _size);
	return true;
}

bool CBAWMeshFileLoader::decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const
{
	SizeT dstSize = _dstSize;
	SizeT srcSize = _srcSize - LZMA_PROPS_SIZE;
	ELzmaStatus status;
	ISzAlloc alloc{&asset::LzmaMemMngmnt::alloc, &asset::LzmaMemMngmnt::release};
	const SRes res = LzmaDecode((Byte*)_dst, &dstSize, (const Byte*)(_src)+LZMA_PROPS_SIZE, &srcSize, (const Byte*)_src, LZMA_PROPS_SIZE, LZMA_FINISH_ANY, &status, &alloc);
	if (res != SZ_OK)
		return false;
	return true;
}

bool CBAWMeshFileLoader::decompressLz4(void * _dst, size_t _dstSize, const void * _src, size_t _srcSize) const
{
	int res = LZ4_decompress_safe((const char*)_src, (char*)_dst, _srcSize, _dstSize);
	return res >= 0;
}

}} // irr::scene
