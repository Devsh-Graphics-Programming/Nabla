// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_TEXTURE_PATH_BLOB_H_INCLUDED__
#define __IRR_TEXTURE_PATH_BLOB_H_INCLUDED__

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

template<>
<<<<<<< HEAD
struct CorrespondingBlobTypeFor<video::IRenderableVirtualTexture> { typedef TexturePathBlobV1 type; };
=======
struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef TexturePathBlobV2 type; };
>>>>>>> 4b8849c91cc8553bc1cb4ac3113119480f61b467

}
} // irr::asset
#endif

#endif
