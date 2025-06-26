// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_WRITER_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_WRITER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IAssetWriter.h"


namespace nbl::asset
{

class IGeometryWriter : public IAssetWriter
{
	public:
		virtual inline uint64_t getSupportedAssetTypesBitfield() const override {return IAsset::ET_GEOMETRY;}

	protected:
		IGeometryWriter() {}
		virtual ~IGeometryWriter() = 0;

	private:
};

}

#endif
