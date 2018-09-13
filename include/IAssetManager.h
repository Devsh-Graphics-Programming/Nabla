// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef _I_ASSET_MANAGER_H_INCLUDED_
#define _I_ASSET_MANAGER_H_INCLUDED_

// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IFileSystem.h"
#include "CConcurrentObjectCache.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "CGeometryCreator.h"

#define USE_MAPS_FOR_PATH_BASED_CACHE //benchmark and choose, paths can be full system paths

namespace irr
{
	namespace asset
	{
		typedef scene::ICPUMesh ICPUMesh;

		class IAssetManager// : public core::IReferenceCounted
		{
			public:
				#ifdef USE_MAPS_FOR_PATH_BASED_CACHE
					typedef core::CConcurrentObjectCache<std::string, asset::ICPUMesh, core::map> MeshCache_T;
				#else
					typedef core::CConcurrentObjectCache<std::string, asset::ICPUMesh, core::vector> MeshCache_T;
				#endif // USE_MAPS_FOR_PATH_BASED_CACHE

			public:
				virtual const MeshCache_T& getMeshCache() const = 0;
		};
	}
}

#endif
