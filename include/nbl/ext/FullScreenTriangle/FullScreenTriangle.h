// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_
#define _NBL_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_

#include "nabla.h"

namespace nbl::ext::FullScreenTriangle
{
struct ProtoPipeline final
{
	public:
		static core::smart_refctd_ptr<asset::IShader> createDefaultVertexShader(asset::IAssetManager* assMan, video::ILogicalDevice* device, system::ILogger* logger=nullptr);
		static core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias = "nbl/ext/FullScreenTriangle");

		ProtoPipeline(asset::IAssetManager* assMan, video::ILogicalDevice* device, system::ILogger* logger=nullptr);

		operator bool() const;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(
			const video::IGPUPipelineBase::SShaderSpecInfo& fragShader,
			video::IGPUPipelineLayout* layout,
			const video::IGPURenderpass* renderpass,
			const uint32_t subpassIx=0,
			asset::SBlendParams blendParams = {},
			const hlsl::SurfaceTransform::FLAG_BITS swapchainTransform=hlsl::SurfaceTransform::FLAG_BITS::IDENTITY_BIT
		);

		core::smart_refctd_ptr<asset::IShader> m_vxShader;
};

bool recordDrawCall(video::IGPUCommandBuffer* commandBuffer);
}
#endif
