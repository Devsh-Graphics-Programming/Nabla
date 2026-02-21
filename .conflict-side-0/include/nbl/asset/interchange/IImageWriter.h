// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_IMAGE_WRITER_H_INCLUDED_
#define _NBL_ASSET_I_IMAGE_WRITER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUImageView.h"

#include "nbl/asset/interchange/IImageAssetHandlerBase.h"
#include "nbl/asset/interchange/IAssetWriter.h"

#include "nbl/asset/filters/CFlattenRegionsImageFilter.h"


namespace nbl::asset
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
#endif
