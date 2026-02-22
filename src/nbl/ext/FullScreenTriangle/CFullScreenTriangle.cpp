// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/ext/FullScreenTriangle/builtin/build/CArchive.h"
#endif

#include "nbl/ext/FullScreenTriangle/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;

namespace nbl::ext::FullScreenTriangle
{

constexpr std::string_view NBL_EXT_MOUNT_ENTRY = "nbl/ext/FullScreenTriangle";
constexpr std::string_view VertexEntryPoint = "__nbl__hlsl__ext__FullScreenTriangle__vertex_main";

smart_refctd_ptr<IFileArchive> ProtoPipeline::mount(smart_refctd_ptr<ILogger> logger, ISystem* system, ILogicalDevice* device, const std::string_view archiveAlias)
{
	assert(system);
	if (!system)
		return nullptr;

	const auto composed = path(archiveAlias.data()) / builtin::build::get_spirv_key<"full_screen_triangle_vertex">(device);
	if (system->exists(composed, {}))
		return nullptr;

#ifdef NBL_EMBED_BUILTIN_RESOURCES
	auto archive = make_smart_refctd_ptr<builtin::build::CArchive>(smart_refctd_ptr(logger));
#else
	auto archive = make_smart_refctd_ptr<CMountDirectoryArchive>(std::string_view(NBL_FULL_SCREEN_TRIANGLE_HLSL_MOUNT_POINT), smart_refctd_ptr(logger), system);
#endif

	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
	return smart_refctd_ptr(archive);
}

smart_refctd_ptr<IShader> ProtoPipeline::createDefaultVertexShader(IAssetManager* assMan, ILogicalDevice* device, ILogger* logger)
{
	if (!assMan || !device)
		return nullptr;

	auto system = smart_refctd_ptr<ISystem>(assMan->getSystem());
	if (system)
		ProtoPipeline::mount(smart_refctd_ptr<ILogger>(logger), system.get(), device, NBL_EXT_MOUNT_ENTRY);

	IAssetLoader::SAssetLoadParams lp = {};
	lp.logger = logger;
	lp.workingDirectory = NBL_EXT_MOUNT_ENTRY.data();

	const auto key = builtin::build::get_spirv_key<"full_screen_triangle_vertex">(device);
	auto bundle = assMan->getAsset(key.c_str(), lp);
	const auto assets = bundle.getContents();
	if (assets.empty())
		return nullptr;

	auto source = IAsset::castDown<IShader>(assets[0]);
	if (!source)
		return nullptr;

	return device->compileShader({.source = source.get(), .stage = hlsl::ESS_VERTEX});
}

ProtoPipeline::ProtoPipeline(IAssetManager* assMan, ILogicalDevice* device, ILogger* logger)
{
	m_vxShader = createDefaultVertexShader(assMan, device, logger);
}

ProtoPipeline::operator bool() const
{
	return m_vxShader.get();
}

smart_refctd_ptr<IGPUGraphicsPipeline> ProtoPipeline::createPipeline(
	const IGPUPipelineBase::SShaderSpecInfo& fragShader,
	IGPUPipelineLayout* layout,
	const IGPURenderpass* renderpass,
	const uint32_t subpassIx,
	SBlendParams blendParams,
	const hlsl::SurfaceTransform::FLAG_BITS swapchainTransform)
{
	if (!renderpass || !bool(*this) || hlsl::bitCount(swapchainTransform) != 1)
		return nullptr;

	auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());

	smart_refctd_ptr<IGPUGraphicsPipeline> m_retval;
	{
		constexpr SRasterizationParams defaultRasterParams = {
			.faceCullingMode = EFCM_NONE,
			.depthWriteEnable = false,
			.depthCompareOp = ECO_ALWAYS
		};
		const auto orientationAsUint32 = static_cast<uint32_t>(swapchainTransform);

		IGPUPipelineBase::SShaderEntryMap specConstants;
		specConstants[0] = std::span{ reinterpret_cast<const uint8_t*>(&orientationAsUint32), sizeof(orientationAsUint32) };

		IGPUGraphicsPipeline::SCreationParams params[1];
		params[0].layout = layout;
		params[0].vertexShader = { .shader = m_vxShader.get(), .entryPoint = VertexEntryPoint.data(), .entries = &specConstants };
		params[0].fragmentShader = fragShader;
		params[0].cached = {
			.vertexInput = {}, // The Full Screen Triangle doesn't use any HW vertex input state
			.primitiveAssembly = {},
			.rasterization = defaultRasterParams,
			.blend = blendParams,
			.subpassIx = subpassIx
		};
		params[0].renderpass = renderpass;

		if (!device->createGraphicsPipelines(nullptr, params, &m_retval))
			return nullptr;
	}
	return m_retval;
}

bool recordDrawCall(IGPUCommandBuffer* commandBuffer)
{
	constexpr auto VERTEX_COUNT = 3;
	constexpr auto INSTANCE_COUNT = 1;
	return commandBuffer->draw(VERTEX_COUNT, INSTANCE_COUNT, 0, 0);
}

}
