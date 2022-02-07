// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/bawformat/CBlobsLoadingManager.h"

#include "nbl/asset/bawformat/CBAWFile.h"
#include "ISceneManager.h"
#include "IFileSystem.h"

//! Adds support of given blob type to BlobsLoadingManager. For use ONLY inside BlobsLoadingManager's member functions. _NBL_SUPPORTED_BLOBS is defined in IrrCompileConfig.h.
#define _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(Function, BlobType, ...) \
    switch(BlobType)                                                       \
    {                                                                      \
        _NBL_SUPPORTED_BLOBS(Function, __VA_ARGS__)                        \
    }

namespace nbl
{
namespace asset
{
core::unordered_set<uint64_t> CBlobsLoadingManager::getNeededDeps(uint32_t _blobType, const void* _blob)
{
#ifdef OLD_SHADERS
    _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(getNeededDeps, _blobType, _blob)
#endif
    return core::unordered_set<uint64_t>();
}

void* CBlobsLoadingManager::instantiateEmpty(uint32_t _blobType, const void* _blob, size_t _blobSize, BlobLoadingParams& _params)
{
#ifdef OLD_SHADERS
    _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(instantiateEmpty, _blobType, _blob, _blobSize, _params)
#endif
    return nullptr;
}

void* CBlobsLoadingManager::finalize(uint32_t _blobType, void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params)
{
#ifdef OLD_SHADERS
    _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(finalize, _blobType, _obj, _blob, _blobSize, _deps, _params)
#endif
    return nullptr;
}

void CBlobsLoadingManager::releaseObj(uint32_t _blobType, void* _obj)
{
#ifdef OLD_SHADERS
    _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(releaseObj, _blobType, _obj)
#endif
}

/*
inline core::string memberPackingDebugSupportFunc(uint32_t _blobType)
{
    _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER(printMemberPackingDebug, _blobType)
}

void CBlobsLoadingManager::printMemberPackingDebug()
{
    for (uint32_t blobType=EBT_MESH; blobType<EBT_COUNT; blobType++)
        printf("%s\n",memberPackingDebugSupportFunc(blobType));
}*/

#undef _NBL_GENERAL_BLOB_FUNCTION_SWITCH_WRAPPER

}
}  // nbl::asset
