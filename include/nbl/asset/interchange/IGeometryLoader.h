// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_LOADER_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_LOADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"


namespace nbl::asset
{

class IGeometryLoader : public IAssetLoader
{
	public:
		virtual inline uint64_t getSupportedAssetTypesBitfield() const override {return IAsset::ET_GEOMETRY;}

	protected:
		IGeometryLoader() {}
		virtual ~IGeometryLoader() = 0;

	private:
};

}

#endif
