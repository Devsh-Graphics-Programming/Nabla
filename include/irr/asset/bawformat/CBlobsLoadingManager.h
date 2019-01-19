// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BLOBS_LOADING_MANAGER_H_INCLUDED__
#define __IRR_BLOBS_LOADING_MANAGER_H_INCLUDED__

#include <unordered_map>

#include "stdint.h"
#include "irr/asset/IMesh.h"
#include "path.h"
#include "irr/asset/IAssetLoader.h"

namespace irr {

class IrrlichtDevice;

namespace scene
{
	class ISceneManager;
	class CFinalBoneHierarchy;
}
namespace asset
{
    class ICPUSkinnedMesh;
    class ICPUSkinnedMeshBuffer;
}
namespace io
{
	class IFileSystem;
}

namespace asset
{

	struct BlobLoadingParams
	{
        IAssetLoader* ldr;
        IrrlichtDevice* device;
		io::IFileSystem* fs;
		io::path filePath;
        asset::IAssetLoader::SAssetLoadParams params;
        asset::IAssetLoader::IAssetLoaderOverride* loaderOverride;
	};

	//! Class abstracting blobs version from process of loading them from *.baw file.
	/**
	If you wish to extend .baw format by your own blob types, here's what you need to do:
		- Add a corresponding to your blob value to Blob::E_BLOB_TYPE enum
		- Make a class (or struct, a matter of keyword) representing your blob
		- Your class must inherit from irr::core::TypedBlob<BlobType, Type> and define specialization of its member functions (see blobsLoading.cpp for existing code as an example):
			-# `core::unordered_set<uint64_t> getNeededDeps(const void* _blob);` - returns vector of handles to dependency blobs
			-# `void* instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params);` - instantiates (i.e. dynamically by `new` allocates) an object without creating any possible dependent objects that have to be loaded from file as another blob
			-# `void* finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params);` - finalizes the object assigning any dependent object to appropriate field of the object being finalized
			-# `void releaseObj(const void* _obj);` - destroys given object
		- Let `BlobsLoadingManager` know about your blob type, by editing _IRR_SUPPORTED_BLOBS accesible in IrrCompileConfig.h.

	Feature not ready yet. (only loading actually)
	*/
	class IRR_FORCE_EBO CBlobsLoadingManager
	{
	public:
		core::unordered_set<uint64_t> getNeededDeps(uint32_t _blobType, const void* _blob);
		void* instantiateEmpty(uint32_t _blobType, const void* _blob, size_t _blobSize, const BlobLoadingParams& _params);
		void* finalize(uint32_t _blobType, void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params);
		void releaseObj(uint32_t _blobType, void* _obj);

		//static void printMemberPackingDebug();
	};

}} // irr::asset

#endif
