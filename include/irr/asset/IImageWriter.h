// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_IMAGE_WRITER_H_INCLUDED__
#define __NBL_ASSET_I_IMAGE_WRITER_H_INCLUDED__

#include "IImage.h"
#include "irr/core/core.h"

#include "irr/asset/IAssetWriter.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/IImageAssetHandlerBase.h"

#include "irr/asset/filters/CFlattenRegionsImageFilter.h"

namespace irr
{
namespace asset
{

class IImageWriter : public IAssetWriter, public IImageAssetHandlerBase
{
	public:

	protected:

		IImageWriter() = default;
		virtual ~IImageWriter() = 0;

	private:
};

}
}

#endif
