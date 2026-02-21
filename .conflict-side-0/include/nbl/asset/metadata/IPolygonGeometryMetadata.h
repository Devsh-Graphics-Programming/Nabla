// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_POLYGON_GEOMETRY_METADATA_H_INCLUDED_
#define _NBL_ASSET_I_POLYGON_GEOMETRY_METADATA_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"


namespace nbl::asset
{

//! 
class IPolygonGeometryMetadata : public core::Interface
{
	public:
		inline IPolygonGeometryMetadata() = default;

	protected:
		virtual ~IPolygonGeometryMetadata() = default;

		inline IPolygonGeometryMetadata& operator=(IPolygonGeometryMetadata&& other)
		{
			return *this;
		}
};

}
#endif
