// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_IMAGE_LOADER_H_INCLUDED_
#define _NBL_ASSET_I_IMAGE_LOADER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

namespace nbl::asset
{

class IImageLoader : public IAssetLoader, public IImageAssetHandlerBase
{
	public:

	protected:

		IImageLoader() {}
		virtual ~IImageLoader() = 0;

	private:
};

}

#endif
