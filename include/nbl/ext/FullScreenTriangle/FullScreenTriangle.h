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
		inline core::smart_refctd_ptr<asset::IShader> createDefaultVertexShader(asset::IAssetManager* assMan, video::ILogicalDevice* device, system::ILogger* logger=nullptr)
		{
			if (!assMan || !device)
				return nullptr;
	
			using namespace ::nbl::asset;
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = logger;
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = assMan->getAsset("nbl/builtin/hlsl/ext/FullScreenTriangle/default.vert.hlsl",lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return nullptr;

			auto source = IAsset::castDown<IShader>(assets[0]);
			if (!source)
				return nullptr;

			return device->compileShader({ .source = source.get(), .stage = hlsl::ESS_VERTEX });
		}

	public:
		inline ProtoPipeline(asset::IAssetManager* assMan, video::ILogicalDevice* device, system::ILogger* logger=nullptr)
		{
			m_vxShader = createDefaultVertexShader(assMan,device,logger);
			m_vxEntryPoint = "main";
		}

		inline ProtoPipeline(core::smart_refctd_ptr<asset::IShader> vertexShader, const char* vertexEntryPoint="main") : m_vxShader(std::move(vertexShader))
		{
			m_vxEntryPoint = vertexEntryPoint ? vertexEntryPoint : "main";
		}

		inline operator bool() const {return m_vxShader.get();}

		inline core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(
			const video::IGPUPipelineBase::SShaderSpecInfo& fragShader,
			video::IGPUPipelineLayout* layout,
			const video::IGPURenderpass* renderpass,
			const uint32_t subpassIx=0,
			asset::SBlendParams blendParams = {},
			const hlsl::SurfaceTransform::FLAG_BITS swapchainTransform=hlsl::SurfaceTransform::FLAG_BITS::IDENTITY_BIT
		)
		{
			if (!renderpass || !bool(*this) || hlsl::bitCount(swapchainTransform)!=1)
				return nullptr;

			using namespace ::nbl::video;
			auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());

			core::smart_refctd_ptr<IGPUGraphicsPipeline> m_retval;
			{
				const auto orientationAsUint32 = static_cast<uint32_t>(swapchainTransform);

        IGPUPipelineBase::SShaderEntryMap specConstants;
				specConstants[0] = std::span{ reinterpret_cast<const uint8_t*>(&orientationAsUint32), sizeof(orientationAsUint32)};

				IGPUGraphicsPipeline::SCreationParams params[1];
				params[0].layout = layout;
				params[0].vertexShader = { .shader = m_vxShader.get(), .entryPoint = m_vxEntryPoint, .entries = &specConstants };
				params[0].fragmentShader = fragShader;
				params[0].cached = {
					.vertexInput = {}, // The Full Screen Triangle doesn't use any HW vertex input state
					.primitiveAssembly = {},
					.rasterization = DefaultRasterParams,
					.blend = blendParams,
					.subpassIx = subpassIx
				};
				params[0].renderpass = renderpass;

				if (!device->createGraphicsPipelines(nullptr,params,&m_retval))
					return nullptr;
			}
			return m_retval;
		}


		core::smart_refctd_ptr<asset::IShader> m_vxShader;
		std::string m_vxEntryPoint = "main";
		// The default is correct for us
		constexpr static inline asset::SRasterizationParams DefaultRasterParams = {
			.faceCullingMode = asset::EFCM_NONE,
			.depthWriteEnable = false,
			.depthCompareOp = asset::ECO_ALWAYS
		};
};

	
/*
	Helper function for drawing full screen triangle.
	It should be called between command buffer render pass
	records.
*/
static inline bool recordDrawCall(video::IGPUCommandBuffer* commandBuffer)
{
	constexpr auto VERTEX_COUNT = 3;
	constexpr auto INSTANCE_COUNT = 1;
	return commandBuffer->draw(VERTEX_COUNT,INSTANCE_COUNT,0,0);
}
}
#endif
