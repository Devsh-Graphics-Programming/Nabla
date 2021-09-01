// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_
#define _NBL_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_

#include "nabla.h"

namespace nbl
{
	namespace ext
	{
		namespace FullScreenTriangle
		{
			using NBL_PROTO_PIPELINE = std::tuple<core::smart_refctd_ptr<video::IGPUSpecializedShader>, asset::SVertexInputParams, asset::SPrimitiveAssemblyParams, asset::SBlendParams, nbl::asset::SRasterizationParams>;

			inline NBL_PROTO_PIPELINE createProtoPipeline(nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams)
			{
				if (!cpu2gpuParams.assetManager)
					assert(false);

				nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
				auto* assetManager = cpu2gpuParams.assetManager;

				NBL_PROTO_PIPELINE protoPipeline;

				asset::IAsset::E_TYPE types[] = { asset::IAsset::ET_SPECIALIZED_SHADER,static_cast<asset::IAsset::E_TYPE>(0u) };
				auto found = assetManager->findAssets("nbl/builtin/specialized_shader/fullscreentriangle.vert", types);
				assert(found->size());
				auto contents = found->begin()->getContents();
				assert(!contents.empty());
				auto pShader = static_cast<asset::ICPUSpecializedShader*>((contents.begin()->get()));

				auto& gpuSpecializedShader = std::get<core::smart_refctd_ptr<video::IGPUSpecializedShader>>(protoPipeline);
				{
					auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&pShader, &pShader + 1, cpu2gpuParams);
					if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
						assert(false);

					gpuSpecializedShader = (*gpu_array)[0];
				}

				auto& inputParams = std::get<asset::SVertexInputParams>(protoPipeline);
				{
					inputParams.enabledBindingFlags = inputParams.enabledAttribFlags = 0u;
					for (size_t i=0ull; i<asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; i++)
						inputParams.attributes[i] = {0u,asset::EF_UNKNOWN,0u};
					for (size_t i=0ull; i<asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
						inputParams.bindings[i] = {0u,asset::EVIR_PER_VERTEX};
				}

				auto& assemblyParams = std::get<asset::SPrimitiveAssemblyParams>(protoPipeline);
				assemblyParams.primitiveRestartEnable = false;
				assemblyParams.primitiveType = asset::EPT_TRIANGLE_LIST;
				assemblyParams.tessPatchVertCount = 3u;

				auto& blendParams = std::get<asset::SBlendParams>(protoPipeline);
				blendParams.logicOpEnable = false;
				blendParams.logicOp = nbl::asset::ELO_NO_OP;
				for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
					blendParams.blendParams[i].attachmentEnabled = (i == 0ull);

				auto& rasterParams = std::get<asset::SRasterizationParams>(protoPipeline);
				rasterParams.faceCullingMode = nbl::asset::EFCM_NONE;
				rasterParams.depthCompareOp = nbl::asset::ECO_ALWAYS;
				rasterParams.minSampleShading = 1.f;
				rasterParams.depthWriteEnable = false;
				rasterParams.depthTestEnable = false;

				return protoPipeline;
			}

			inline core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline(video::ILogicalDevice* logicalDevice, NBL_PROTO_PIPELINE& protoPipeline, core::smart_refctd_ptr<video::IGPUSpecializedShader>&& gpuFragmentShader, core::smart_refctd_ptr<video::IGPUPipelineLayout>&& pipelineLayout)
			{
				if (!logicalDevice)
					assert(false);

				video::IGPUSpecializedShader* gpuShaders[] = { std::get<core::smart_refctd_ptr<video::IGPUSpecializedShader>>(protoPipeline).get(), gpuFragmentShader.get() };
				auto gpuRenderpassIndependentPipeline = logicalDevice->createGPURenderpassIndependentPipeline
				(	
					nullptr,
					std::move(pipelineLayout),
					gpuShaders,
					gpuShaders + 2,
					std::get<asset::SVertexInputParams>(protoPipeline),
					std::get<asset::SBlendParams>(protoPipeline),
					std::get<asset::SPrimitiveAssemblyParams>(protoPipeline),
					std::get<asset::SRasterizationParams>(protoPipeline)
				);

				return gpuRenderpassIndependentPipeline;
			}

			/*
				Helper function for drawing full screen triangle.
				It should be called between command buffer render pass
				records.
			*/

			inline bool recordDrawCalls(video::IGPUCommandBuffer* commandBuffer)
			{
				_NBL_STATIC_INLINE_CONSTEXPR auto VERTEX_COUNT = 3;
				_NBL_STATIC_INLINE_CONSTEXPR auto INSTANCE_COUNT = 1;

				const nbl::video::IGPUBuffer* gpuBufferBindings[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
				{
					for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
						gpuBufferBindings[i] = nullptr;
				}

				size_t bufferBindingsOffsets[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
				{
					for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
						bufferBindingsOffsets[i] = 0;
				}

				commandBuffer->bindVertexBuffers(0, nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, gpuBufferBindings, bufferBindingsOffsets);
				commandBuffer->bindIndexBuffer(nullptr, 0, nbl::asset::EIT_UNKNOWN);

				return commandBuffer->draw(VERTEX_COUNT, INSTANCE_COUNT, 0, 0);
			}
		}
	}
}

#endif

