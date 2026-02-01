// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_COLLECTION_METADATA_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_COLLECTION_METADATA_H_INCLUDED_


#include "nbl/asset/ICPUGeometryCollection.h"


namespace nbl::asset
{

//! 
class IGeometryCollectionMetadata : public core::Interface
{
	public:
		inline IGeometryCollectionMetadata() = default;

	protected:
		virtual ~IGeometryCollectionMetadata() = default;

		inline IGeometryCollectionMetadata& operator=(IGeometryCollectionMetadata&& other)
		{
			return *this;
		}
};

}
#endif
