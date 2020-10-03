// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#ifndef __NBL_TEXTURE_PATH_BLOB_H_INCLUDED__
#define __NBL_TEXTURE_PATH_BLOB_H_INCLUDED__

//! kill this whole file soon (upgrade BaW format to V3)
#ifndef NEW_SHADERS
#include "irr/asset/ICPUTexture.h"

namespace irr
{
namespace video
{
	class IRenderableVirtualTexture;
}
namespace asset
{

#include "irr/irrpack.h"
struct IRR_FORCE_EBO TexturePathBlobV0 : TypedBlob<TexturePathBlobV0, ICPUTexture>, VariableSizeBlob<TexturePathBlobV0, ICPUTexture>
{};
#include "irr/irrunpack.h"

using TexturePathBlobV1 = TexturePathBlobV0;
using TexturePathBlobV2 = TexturePathBlobV1;
using TexturePathBlobV3 = TexturePathBlobV2;

template<>
struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef TexturePathBlobV3 type; };

}
} // irr::asset
#endif

#endif
