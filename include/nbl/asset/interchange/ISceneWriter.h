// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SCENE_WRITER_H_INCLUDED_
#define _NBL_ASSET_I_SCENE_WRITER_H_INCLUDED_
#include "nbl/core/declarations.h"
#include "nbl/asset/ICPUScene.h"
#include "nbl/asset/interchange/IAssetWriter.h"
namespace nbl::asset
{
//! Writer base for exporters whose root asset type is `ET_SCENE`.
class ISceneWriter : public IAssetWriter
{
	public:
		virtual inline uint64_t getSupportedAssetTypesBitfield() const override { return IAsset::ET_SCENE; }
	protected:
		ISceneWriter() = default;
		virtual ~ISceneWriter() = default;
};
}
#endif
