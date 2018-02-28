// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBlobsLoadingManager.h"

#include "CBAWFile.h"
#include "ISceneManager.h"
#include "IFileSystem.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"

//! Adds support of given blob type to BlobsLoadingManager. For use ONLY inside BlobsLoadingManager's member functions. _IRR_SUPPORTED_BLOBS is defined in IrrCompileConfig.h.
#define _IRR_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(Function, BlobType, ...)\
switch(BlobType)\
{\
_IRR_SUPPORTED_BLOBS(Function, __VA_ARGS__)\
}

namespace irr { namespace core
{
std::vector<uint64_t> CBlobsLoadingManager::getNeededDeps(uint32_t _blobType, const void * _blob)
{
	_IRR_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(getNeededDeps, _blobType, _blob)
	return std::vector<uint64_t>();
}

void* CBlobsLoadingManager::instantiateEmpty(uint32_t _blobType, const void* _blob, size_t _blobSize, const BlobLoadingParams& _params)
{
	_IRR_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(instantiateEmpty, _blobType, _blob, _blobSize, _params)
	return NULL;
}

void* CBlobsLoadingManager::finalize(uint32_t _blobType, void* _obj, const void* _blob, size_t _blobSize, std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params)
{
	_IRR_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(finalize, _blobType, _obj, _blob, _blobSize, _deps, _params)
	return NULL;
}

void CBlobsLoadingManager::releaseObj(uint32_t _blobType, void * _obj)
{
	_IRR_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(releaseObj, _blobType, _obj)
}

#undef _IRR_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER

}} // irr::core