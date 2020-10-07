// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __IRR_I_IMAGE_LOADER_H_INCLUDED__
#define __IRR_I_IMAGE_LOADER_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/IAssetLoader.h"
#include "irr/asset/IImageAssetHandlerBase.h"
#include "irr/asset/ICPUImageView.h"

namespace irr
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

#endif // __IRR_I_IMAGE_LOADER_H_INCLUDED__
