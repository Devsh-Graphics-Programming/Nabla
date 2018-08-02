// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef _C_ASSET_MANAGER_H_INCLUDED_
#define _C_ASSET_MANAGER_H_INCLUDED_


#include "IAssetManager.h"

namespace irr
{
	namespace asset
	{
		class CAssetManager : public IAssetManager
		{
			protected:
				io::IFileSystem* fileSystem;

				MeshCache_T meshCache;
			public:
				CAssetManager(io::IFileSystem* fs);

				virtual const MeshCache_T& getMeshCache() const { return meshCache; }

				virtual const ICPUMesh* getMesh(std::string& path) const { return meshCache.getByKey(path); }
		};
	}
}

#endif