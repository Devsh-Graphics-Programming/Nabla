// Copyright (C) 2019-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_C_STL_MESH_FILE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_STL_MESH_FILE_LOADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/interchange/IGeometryLoader.h"


namespace nbl::asset
{

//! Meshloader capable of loading STL meshes.
class CSTLMeshFileLoader final : public IGeometryLoader
{
	public:
		explicit CSTLMeshFileLoader(asset::IAssetManager* _assetManager);

		asset::SAssetBundle loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override;
};

}	// end namespace nbl::scene
#endif

