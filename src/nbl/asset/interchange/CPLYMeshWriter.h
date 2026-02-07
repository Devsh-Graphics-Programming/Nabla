// Copyright (C) 2019-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_PLY_MESH_WRITER_H_INCLUDED_
#define _NBL_ASSET_PLY_MESH_WRITER_H_INCLUDED_


#include "nbl/asset/interchange/IGeometryWriter.h"


namespace nbl::asset
{

//! class to write PLY mesh files
class CPLYMeshWriter : public IGeometryWriter
{
	public:
		CPLYMeshWriter();

		const char** getAssociatedFileExtensions() const override;

		uint32_t getSupportedFlags() override;
		uint32_t getForcedFlags() override;

		bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

} // end namespace
#endif
