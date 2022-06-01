// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_BLOBS_LOADING_MANAGER_H_INCLUDED__
#define __NBL_ASSET_BLOBS_LOADING_MANAGER_H_INCLUDED__

#include <unordered_map>

#include "stdint.h"

#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/interchange/IAssetLoader.h"

#if 0
namespace nbl::asset
{
    class ICPUSkinnedMesh;
    class ICPUSkinnedMeshBuffer;
	class IAssetManager;


	struct NBL_API BlobLoadingParams
	{
        IAssetLoader* const ldr;
        IAssetManager* const manager;
		io::IFileSystem* const fs;
		const std::filesystem::path filePath;
        const asset::IAssetLoader::SAssetLoadParams params;
        asset::IAssetLoader::IAssetLoaderOverride* const loaderOverride;
		core::stack<core::smart_refctd_ptr<ICPUMesh>> meshesToFlip;
	};

	//! Class abstracting blobs version from process of loading them from *.baw file.
	/**
	If you wish to extend .baw format by your own blob types, here's what you need to do:
		- Add a corresponding to your blob value to Blob::E_BLOB_TYPE enum
		- Make a class (or struct, a matter of keyword) representing your blob
		- Your class must inherit from nbl::core::TypedBlob<BlobType, Type> and define specialization of its member functions (see blobsLoading.cpp for existing code as an example):
			-# `core::unordered_set<uint64_t> getNeededDeps(const void* _blob);` - returns vector of handles to dependency blobs
			-# `void* instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params);` - instantiates (i.e. dynamically by `new` allocates) an object without creating any possible dependent objects that have to be loaded from file as another blob
			-# `void* finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params);` - finalizes the object assigning any dependent object to appropriate field of the object being finalized
			-# `void releaseObj(const void* _obj);` - destroys given object
		- Let `BlobsLoadingManager` know about your blob type, by editing _NBL_SUPPORTED_BLOBS accesible in IrrCompileConfig.h.

	Feature not ready yet. (only loading actually)
	*/
	class NBL_API NBL_FORCE_EBO CBlobsLoadingManager
	{
	public:
		core::unordered_set<uint64_t> getNeededDeps(uint32_t _blobType, const void* _blob);
		void* instantiateEmpty(uint32_t _blobType, const void* _blob, size_t _blobSize, BlobLoadingParams& _params);
		void* finalize(uint32_t _blobType, void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params);
		void releaseObj(uint32_t _blobType, void* _obj);

		//static void printMemberPackingDebug();
	};

} // nbl::asset

#endif
#endif
