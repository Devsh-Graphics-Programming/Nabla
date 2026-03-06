// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_OBJ_MESH_WRITER_H_INCLUDED_
#define _NBL_ASSET_OBJ_MESH_WRITER_H_INCLUDED_

#include "nbl/asset/interchange/ISceneWriter.h"

namespace nbl::asset
{
/*
	Writes OBJ from a single polygon geometry, a geometry collection, or a scene.
	OBJ itself is still treated here as final flattened geometry data, not as a scene format.
	Scene input is accepted only as export input: the writer bakes transforms and serializes all collected polygon geometries into one OBJ stream.
	This preserves the final shape but does not try to keep scene-only structure such as hierarchy or instancing.
	In other words `ET_SCENE -> OBJ` is supported as flattening, not as round-tripping scene semantics through the OBJ format.
*/
class COBJMeshWriter : public ISceneWriter
{
	public:
		COBJMeshWriter();

		uint64_t getSupportedAssetTypesBitfield() const override;

		const char** getAssociatedFileExtensions() const override;

		writer_flags_t getSupportedFlags() override;

		writer_flags_t getForcedFlags() override;

		bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

} // end namespace

#endif