// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_TEXTURE_PATH_BLOB_H_INCLUDED__
#define __IRR_TEXTURE_PATH_BLOB_H_INCLUDED__

//! kill this whole file soon

namespace irr
{
namespace video
{
	class IVirtualTexture;
}
namespace asset
{

#include "irr/irrpack.h"
struct IRR_FORCE_EBO TexturePathBlobV0 : TypedBlob<TexturePathBlobV0, asset::ICPUTexture>, FixedSizeBlob<TexturePathBlobV0, asset::ICPUTexture>
{};
#include "irr/irrunpack.h"

using TexturePathBlobV1 = TexturePathBlobV0;

template<>
struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef TexturePathBlobV1 type; };

}
} // irr::asset

#endif
