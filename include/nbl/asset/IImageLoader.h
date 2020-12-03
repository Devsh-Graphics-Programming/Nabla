// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_IMAGE_LOADER_H_INCLUDED__
#define __NBL_ASSET_I_IMAGE_LOADER_H_INCLUDED__

#include "nbl/core/core.h"

#include "nbl/asset/IAssetLoader.h"
#include "nbl/asset/IImageAssetHandlerBase.h"
#include "nbl/asset/ICPUImageView.h"

namespace nbl
{
namespace asset
{

class IImageLoader : public IAssetLoader, public IImageAssetHandlerBase
{
	public:

	protected:

		IImageLoader() = default;
		virtual ~IImageLoader() = 0;

	private:
};

}
}

#endif
