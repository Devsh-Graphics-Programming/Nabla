// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_C_OBJ_MESH_FILE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_OBJ_MESH_FILE_LOADER_H_INCLUDED_
#include "nbl/core/declarations.h"
#include "nbl/asset/interchange/IGeometryLoader.h"
namespace nbl::asset
{
/**
	Loads plain OBJ as polygon geometry or geometry collections.
	Multiple `o` and `g` blocks mean multiple geometry pieces in one file,
	not a real scene.
	This loader keeps that split as geometry collections because plain OBJ
	does not define scene hierarchy, instancing, or node transforms.
	OBJ/MTL material data also belongs here and remains TODO,
	but that still does not turn plain OBJ into a scene format.
	A single mesh payload can therefore load as one geometry,
	while multiple split pieces still load as geometry collections
	instead of a synthetic scene.

	References:
	- https://www.loc.gov/preservation/digital/formats/fdd/fdd000507
	- https://www.fileformat.info/format/wavefrontobj/egff.htm
*/
class COBJMeshFileLoader : public IGeometryLoader
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
