// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SCENE_LOADER_H_INCLUDED_
#define _NBL_ASSET_I_SCENE_LOADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUScene.h"
#include "nbl/asset/interchange/IAssetLoader.h"


namespace nbl::asset
{

class ISceneLoader : public IAssetLoader
{
	public:
		virtual inline uint64_t getSupportedAssetTypesBitfield() const override {return IAsset::ET_SCENE;}

	protected:
		inline ISceneLoader() {}

	private:
};

}

#endif
