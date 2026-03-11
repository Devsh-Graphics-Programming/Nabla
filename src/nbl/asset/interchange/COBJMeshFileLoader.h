// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_C_OBJ_MESH_FILE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_OBJ_MESH_FILE_LOADER_H_INCLUDED_
#include "nbl/core/declarations.h"
#include "nbl/asset/interchange/ISceneLoader.h"
namespace nbl::asset
{
/**
	Loads plain OBJ into a flat `ICPUScene`.
	Multiple `o` and `g` blocks become separate scene instances backed by
	geometry collections.
	All instance transforms stay identity here.
	Material tables stay invalid until `MTL` support is implemented.

	References:
	- https://www.loc.gov/preservation/digital/formats/fdd/fdd000507
	- https://www.fileformat.info/format/wavefrontobj/egff.htm
*/
class COBJMeshFileLoader : public ISceneLoader
{
	public:
		~COBJMeshFileLoader() override;

		//! Constructor
		explicit COBJMeshFileLoader(IAssetManager* _manager);

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override;

		//! Loads one OBJ asset bundle from an already opened file.
		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};
} // end namespace nbl::asset
#endif
