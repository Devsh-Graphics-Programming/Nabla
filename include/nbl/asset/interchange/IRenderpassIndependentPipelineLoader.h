// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_LOADER_H_INCLUDED__
#define __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_LOADER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl
{
namespace asset
{

class IRenderpassIndependentPipelineLoader : public IAssetLoader
{
	public:
		virtual void initialize() override;

	protected:
		IAssetManager* m_assetMgr;
		core::smart_refctd_dynamic_array<IRenderpassIndependentPipelineMetadata::ShaderInputSemantic> m_basicViewParamsSemantics;

		inline IRenderpassIndependentPipelineLoader(IAssetManager* _am) : m_assetMgr(_am) {}
		virtual ~IRenderpassIndependentPipelineLoader() = 0;

	private:
};

}
}

#endif
