// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_OBJ_MESH_WRITER_H_INCLUDED_
#define _NBL_ASSET_OBJ_MESH_WRITER_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IGeometryWriter.h"


namespace nbl::asset
{

//! class to write OBJ mesh files
class COBJMeshWriter : public IGeometryWriter
{
	public:
		COBJMeshWriter();

		virtual const char** getAssociatedFileExtensions() const
		{
			static const char* ext[]{ "obj", nullptr };
			return ext;
		}

		virtual uint32_t getSupportedFlags() override { return 0u; }

		virtual uint32_t getForcedFlags() { return 0u; }

		virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

} // end namespace

#endif
