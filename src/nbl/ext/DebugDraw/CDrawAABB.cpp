// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/DebugDraw/CDrawAABB.h"

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/ext/debug_draw/builtin/build/CArchive.h"
#endif

#include "nbl/ext/DebugDraw/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;
using namespace hlsl;

namespace nbl::ext::debug_draw
{

core::smart_refctd_ptr<DrawAABB> DrawAABB::create(SCreationParameters&& params)
{
	auto* const logger = params.utilities->getLogger();

	if (!params.validate())
	{
		logger->log("Failed creation parameters validation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	ConstructorParams constructorParams;

	if (params.drawMode & ADM_DRAW_SINGLE)
	{
		auto pipelineLayout = params.singlePipelineLayout;
		if (!pipelineLayout)
			pipelineLayout = createDefaultPipelineLayout(params.utilities->getLogicalDevice(), ADM_DRAW_SINGLE);
		constructorParams.singlePipeline = createPipeline(params, pipelineLayout.get(), ADM_DRAW_SINGLE);
		if (!constructorParams.singlePipeline)
		{
			logger->log("Failed to create pipeline!", ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	if (params.drawMode & ADM_DRAW_BATCH)
	{
		auto pipelineLayout = params.batchPipelineLayout;
		if (!pipelineLayout)
			pipelineLayout = createDefaultPipelineLayout(params.utilities->getLogicalDevice(), ADM_DRAW_BATCH);
		constructorParams.batchPipeline = createPipeline(params, pipelineLayout.get(), ADM_DRAW_BATCH);
		if (!constructorParams.batchPipeline)
		{
			logger->log("Failed to create pipeline!", ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	if (!createStreamingBuffer(params))
	{
		logger->log("Failed to create streaming buffer!", ILogger::ELL_ERROR);
		return nullptr;
	}

	constructorParams.indicesBuffer = createIndicesBuffer(params);
	if (!constructorParams.indicesBuffer)
	{
		logger->log("Failed to create indices buffer!", ILogger::ELL_ERROR);
		return nullptr;
	}

	constructorParams.creationParams = std::move(params);
    return core::smart_refctd_ptr<DrawAABB>(new DrawAABB(std::move(constructorParams)));
}

// extension data mount alias
constexpr std::string_view NBL_EXT_MOUNT_ENTRY = "nbl/ext/DebugDraw";

const smart_refctd_ptr<IFileArchive> DrawAABB::mount(smart_refctd_ptr<ILogger> logger, ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias)
{
	assert(system);

	if (!system)
		return nullptr;

	// the key is deterministic, we are validating presence of required .spv
	const auto composed = path(archiveAlias.data()) / nbl::ext::debug_draw::builtin::build::get_spirv_key<"draw_aabb">(device);
	if (system->exists(composed, {}))
		return nullptr;

	// extension should mount everything for you, regardless if content goes from virtual filesystem 
	// or disk directly - and you should never rely on application framework to expose extension data
	#ifdef NBL_EMBED_BUILTIN_RESOURCES
	auto archive = make_smart_refctd_ptr<builtin::build::CArchive>(smart_refctd_ptr(logger));
	#else
	auto archive = make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(std::string_view(NBL_DEBUG_DRAW_HLSL_MOUNT_POINT), smart_refctd_ptr(logger), system);
	#endif

	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
	return smart_refctd_ptr(archive);
}

smart_refctd_ptr<IGPUGraphicsPipeline> DrawAABB::createPipeline(SCreationParameters& params, const IGPUPipelineLayout* pipelineLayout, DrawMode mode)
{
	system::logger_opt_ptr logger = params.utilities->getLogger();
	auto system = smart_refctd_ptr<ISystem>(params.assetManager->getSystem());
	auto* device = params.utilities->getLogicalDevice();
	mount(smart_refctd_ptr<ILogger>(params.utilities->getLogger()), system.get(), params.utilities->getLogicalDevice(), NBL_EXT_MOUNT_ENTRY);

	auto getShader = [&](const core::string& key)->smart_refctd_ptr<IShader> {
		IAssetLoader::SAssetLoadParams lp = {};
		lp.logger = params.utilities->getLogger();
		lp.workingDirectory = NBL_EXT_MOUNT_ENTRY;
		auto bundle = params.assetManager->getAsset(key.c_str(), lp);

		const auto contents = bundle.getContents();

		if (contents.empty())
		{
			logger.log("Failed to load shader %s from disk", ILogger::ELL_ERROR, key.c_str());
			return nullptr;
		}

		if (bundle.getAssetType() != IAsset::ET_SHADER)
		{
			logger.log("Loaded asset has wrong type!", ILogger::ELL_ERROR);
			return nullptr;
		}

		return IAsset::castDown<IShader>(contents[0]);
	};

	const auto key = nbl::ext::debug_draw::builtin::build::get_spirv_key<"draw_aabb">(device);
	smart_refctd_ptr<IShader> unifiedShader = getShader(key);
	if (!unifiedShader)
	{
		params.utilities->getLogger()->log("Could not compile shaders!", ILogger::ELL_ERROR);
		return nullptr;
	}

	video::IGPUGraphicsPipeline::SCreationParams pipelineParams[1] = {};
	pipelineParams[0].layout = pipelineLayout;
	pipelineParams[0].vertexShader = { .shader = unifiedShader.get(), .entryPoint = (mode & ADM_DRAW_SINGLE) ? "aabb_vertex_single" : "aabb_vertex_instances" };
	pipelineParams[0].fragmentShader = { .shader = unifiedShader.get(), .entryPoint = "aabb_fragment" };
	pipelineParams[0].cached = {
		.primitiveAssembly = {
			.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_LINE_LIST,
		}
	};
	pipelineParams[0].renderpass = params.renderpass.get();

	smart_refctd_ptr<IGPUGraphicsPipeline> pipeline;
	params.utilities->getLogicalDevice()->createGraphicsPipelines(nullptr, pipelineParams, &pipeline);
	if (!pipeline)
	{
		params.utilities->getLogger()->log("Could not create streaming pipeline!", ILogger::ELL_ERROR);
		return nullptr;
	}

	return pipeline;
}

bool DrawAABB::createStreamingBuffer(SCreationParameters& params)
{
	const uint32_t minStreamingBufferAllocationSize = 128u, maxStreamingBufferAllocationAlignment = 4096u, mdiBufferDefaultSize = /* 2MB */ 1024u * 1024u * 2u;

	auto getRequiredAccessFlags = [&](const bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>& properties)
		{
			bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> flags(IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);

			if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
				flags |= IDeviceMemoryAllocation::EMCAF_WRITE;

			return flags;
		};

	if (!params.streamingBuffer)
	{
		IGPUBuffer::SCreationParams mdiCreationParams = {};
		mdiCreationParams.usage = SCachedCreationParameters::RequiredUsageFlags;
		mdiCreationParams.size = mdiBufferDefaultSize;

		auto buffer = params.utilities->getLogicalDevice()->createBuffer(std::move(mdiCreationParams));
		buffer->setObjectDebugName("AABB Streaming Buffer");

		auto memoryReqs = buffer->getMemoryReqs();
		memoryReqs.memoryTypeBits &= params.utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

		auto allocation = params.utilities->getLogicalDevice()->allocate(memoryReqs, buffer.get(), SCachedCreationParameters::RequiredAllocateFlags);
		{
			const bool allocated = allocation.isValid();
			assert(allocated);
		}
		auto memory = allocation.memory;

		if (!memory->map({ 0ull, memoryReqs.size }, getRequiredAccessFlags(memory->getMemoryPropertyFlags())))
			params.utilities->getLogger()->log("Could not map device memory!", ILogger::ELL_ERROR);

		params.streamingBuffer = make_smart_refctd_ptr<SCachedCreationParameters::streaming_buffer_t>(SBufferRange<IGPUBuffer>{0ull, mdiCreationParams.size, std::move(buffer)}, maxStreamingBufferAllocationAlignment, minStreamingBufferAllocationSize);
	}

	auto buffer = params.streamingBuffer->getBuffer();
	auto binding = buffer->getBoundMemory();

	const auto validation = std::to_array
	({
		std::make_pair(buffer->getCreationParams().usage.hasFlags(SCachedCreationParameters::RequiredUsageFlags), "Streaming buffer must be created with IBuffer::EUF_STORAGE_BUFFER_BIT | IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT enabled!"),
		std::make_pair(binding.memory->getAllocateFlags().hasFlags(SCachedCreationParameters::RequiredAllocateFlags), "Streaming buffer's memory must be allocated with IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT enabled!"),
		std::make_pair(binding.memory->isCurrentlyMapped(), "Streaming buffer's memory must be mapped!"), // streaming buffer contructor already validates it, but cannot assume user won't unmap its own buffer for some reason (sorry if you have just hit it)
		std::make_pair(binding.memory->getCurrentMappingAccess().hasFlags(getRequiredAccessFlags(binding.memory->getMemoryPropertyFlags())), "Streaming buffer's memory current mapping access flags don't meet requirements!")
		});

	for (const auto& [ok, error] : validation)
		if (!ok)
		{
			params.utilities->getLogger()->log(error, ILogger::ELL_ERROR);
			return false;
		}

	return true;
}

smart_refctd_ptr<IGPUBuffer> DrawAABB::createIndicesBuffer(SCreationParameters& params)
{
	std::array<uint32_t, IndicesCount> unitAABBIndices;
	unitAABBIndices[0] = 0;
	unitAABBIndices[1] = 1;
	unitAABBIndices[2] = 0;
	unitAABBIndices[3] = 2;

	unitAABBIndices[4] = 3;
	unitAABBIndices[5] = 1;
	unitAABBIndices[6] = 3;
	unitAABBIndices[7] = 2;

	unitAABBIndices[8] = 4;
	unitAABBIndices[9] = 5;
	unitAABBIndices[10] = 4;
	unitAABBIndices[11] = 6;

	unitAABBIndices[12] = 7;
	unitAABBIndices[13] = 5;
	unitAABBIndices[14] = 7;
	unitAABBIndices[15] = 6;

	unitAABBIndices[16] = 0;
	unitAABBIndices[17] = 4;
	unitAABBIndices[18] = 1;
	unitAABBIndices[19] = 5;

	unitAABBIndices[20] = 2;
	unitAABBIndices[21] = 6;
	unitAABBIndices[22] = 3;
	unitAABBIndices[23] = 7;

	auto* device = params.utilities->getLogicalDevice();
	smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
	{
		smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = device->createCommandPool(params.transfer->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf, 1 }))
		{
			params.utilities->getLogger()->log("Failed to create Command Buffer for index buffer!\n");
			return nullptr;
		}
	}

	IGPUBuffer::SCreationParams bufparams;
	bufparams.size = sizeof(uint32_t) * unitAABBIndices.size();
	bufparams.usage = IGPUBuffer::EUF_INDEX_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;

	smart_refctd_ptr<IGPUBuffer> indicesBuffer;
	{
		indicesBuffer = device->createBuffer(std::move(bufparams));
		if (!indicesBuffer)
		{
			params.utilities->getLogger()->log("Failed to create index buffer!\n");
			return nullptr;
		}

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = indicesBuffer->getMemoryReqs();
		reqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		auto bufMem = device->allocate(reqs, indicesBuffer.get());
		if (!bufMem.isValid())
		{
			params.utilities->getLogger()->log("Failed to allocate device memory compatible with index buffer!\n");
			return nullptr;
		}
	}

	{
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdbuf->beginDebugMarker("Fill indices buffer begin");

		SBufferRange<IGPUBuffer> bufRange = { .offset = 0, .size = indicesBuffer->getSize(), .buffer = indicesBuffer };
		cmdbuf->updateBuffer(bufRange, unitAABBIndices.data());

		cmdbuf->endDebugMarker();
		cmdbuf->end();
	}

	smart_refctd_ptr<ISemaphore> idxBufProgress;
	constexpr auto FinishedValue = 25;
	{
		constexpr auto StartedValue = 0;
		idxBufProgress = device->createSemaphore(StartedValue);

		IQueue::SSubmitInfo submitInfos[1] = {};
		const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
		submitInfos[0].commandBuffers = cmdbufs;
		const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = idxBufProgress.get(),.value = FinishedValue,.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS} };
		submitInfos[0].signalSemaphores = signals;

		params.transfer->submit(submitInfos);
	}

	const ISemaphore::SWaitInfo waitInfos[] = { {
				.semaphore = idxBufProgress.get(),
				.value = FinishedValue
			} };
	device->blockForSemaphores(waitInfos);

	return indicesBuffer;
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> DrawAABB::createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange)
{
	return device->createPipelineLayout({ &pcRange , 1 }, nullptr, nullptr, nullptr, nullptr);
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> DrawAABB::createDefaultPipelineLayout(video::ILogicalDevice* device, DrawMode mode)
{
	const uint32_t offset = (mode & ADM_DRAW_BATCH) ? offsetof(ext::debug_draw::PushConstants, ipc) : offsetof(ext::debug_draw::PushConstants, spc);
	const uint32_t pcSize = (mode & ADM_DRAW_BATCH) ? sizeof(SInstancedPC) : sizeof(SSinglePC);
	SPushConstantRange pcRange = {
		.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX,
		.offset = offset,
		.size = pcSize
	};
	return createPipelineLayoutFromPCRange(device, pcRange);
}

bool DrawAABB::renderSingle(const DrawParameters& params, const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t4& color)
{
	if (!(m_cachedCreationParams.drawMode & ADM_DRAW_SINGLE))
	{
		m_cachedCreationParams.utilities->getLogger()->log("DrawAABB has not been enabled for draw single!", ILogger::ELL_ERROR);
		return false;
	}

	auto& commandBuffer = params.commandBuffer;
	commandBuffer->bindGraphicsPipeline(m_singlePipeline.get());
	commandBuffer->setLineWidth(params.lineWidth);
	asset::SBufferBinding<video::IGPUBuffer> indexBinding = { .offset = 0, .buffer = m_indicesBuffer };
	commandBuffer->bindIndexBuffer(indexBinding, asset::EIT_32BIT);

	SSinglePC pc;
	hlsl::float32_t3x4 instanceTransform = getTransformFromAABB(aabb);
	pc.instance.transform = math::linalg::promoted_mul(params.cameraMat, instanceTransform);
	pc.instance.color = color;
	
	commandBuffer->pushConstants(m_singlePipeline->getLayout(), ESS_VERTEX, offsetof(ext::debug_draw::PushConstants, spc), sizeof(SSinglePC), &pc);
	commandBuffer->drawIndexed(IndicesCount, 1, 0, 0, 0);

	return true;
}

hlsl::float32_t3x4 DrawAABB::getTransformFromOBB(const hlsl::shapes::OBB<3, float>& obb)
{
	const auto obbScale = obb.ext * 2.0f;
	const auto axesScaleX = obb.axes[0] * obbScale.x;
	const auto axesScaleY = obb.axes[1] * obbScale.y;
	const auto axesScaleZ = obb.axes[2] * obbScale.z;
	return float32_t3x4{
    axesScaleX.x, axesScaleY.x, axesScaleZ.x, obb.mid.x - (0.5 * (axesScaleX.x + axesScaleY.x + axesScaleZ.x)),
    axesScaleX.y, axesScaleY.y, axesScaleZ.y, obb.mid.y - (0.5 * (axesScaleX.y + axesScaleY.y + axesScaleZ.y)),
    axesScaleX.z, axesScaleY.z, axesScaleZ.z, obb.mid.z - (0.5 * (axesScaleX.z + axesScaleY.z + axesScaleZ.z)),
	};
}

}
