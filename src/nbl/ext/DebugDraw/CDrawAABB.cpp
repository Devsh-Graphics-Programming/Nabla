// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/DebugDraw/CDrawAABB.h"

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/ext/debug_draw/builtin/CArchive.h"
#endif

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

	smart_refctd_ptr<IGPUGraphicsPipeline> singlePipeline = nullptr;
	if (params.drawMode & ADM_DRAW_SINGLE)
	{
		singlePipeline = createPipeline(params, params.singlePipelineLayout.get(), "single.vertex.hlsl", "aabb_instances.fragment.hlsl");
		if (!singlePipeline)
		{
			logger->log("Failed to create pipeline!", ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	smart_refctd_ptr<IGPUGraphicsPipeline> batchPipeline = nullptr;
	if (params.drawMode & ADM_DRAW_BATCH)
	{
		batchPipeline = createPipeline(params, params.batchPipelineLayout.get(), "aabb_instances.vertex.hlsl", "aabb_instances.fragment.hlsl");
		if (!batchPipeline)
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

	auto indicesBuffer = createIndicesBuffer(params);
	if (!indicesBuffer)
	{
		logger->log("Failed to create indices buffer!", ILogger::ELL_ERROR);
		return nullptr;
	}

    return core::smart_refctd_ptr<DrawAABB>(new DrawAABB(std::move(params), singlePipeline, batchPipeline, indicesBuffer));
}

DrawAABB::DrawAABB(SCreationParameters&& params, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> singlePipeline, smart_refctd_ptr<IGPUGraphicsPipeline> batchPipeline, smart_refctd_ptr<IGPUBuffer> indicesBuffer)
    : m_cachedCreationParams(std::move(params)), m_singlePipeline(std::move(singlePipeline)), m_batchPipeline(std::move(batchPipeline)),
    m_indicesBuffer(std::move(indicesBuffer))
{
}

DrawAABB::~DrawAABB()
{
}

// note we use archive entry explicitly for temporary compiler include search path & asset cwd to use keys directly
constexpr std::string_view NBL_ARCHIVE_ENTRY = _ARCHIVE_ENTRY_KEY_;

const smart_refctd_ptr<IFileArchive> DrawAABB::mount(smart_refctd_ptr<ILogger> logger, ISystem* system, const std::string_view archiveAlias)
{
	assert(system);

	if (!system)
		return nullptr;

	// extension should mount everything for you, regardless if content goes from virtual filesystem 
	// or disk directly - and you should never rely on application framework to expose extension data

#ifdef NBL_EMBED_BUILTIN_RESOURCES
	auto archive = make_smart_refctd_ptr<builtin::CArchive>(smart_refctd_ptr(logger));
	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
#else
	auto NBL_EXTENSION_MOUNT_DIRECTORY_ENTRY = (path(_ARCHIVE_ABSOLUTE_ENTRY_PATH_) / NBL_ARCHIVE_ENTRY).make_preferred();
	auto archive = make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(std::move(NBL_EXTENSION_MOUNT_DIRECTORY_ENTRY), smart_refctd_ptr(logger), system);
	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
#endif

	return smart_refctd_ptr(archive);
}

smart_refctd_ptr<IGPUGraphicsPipeline> DrawAABB::createPipeline(SCreationParameters& params, const IGPUPipelineLayout* pipelineLayout, const std::string& vsPath, const std::string& fsPath)
{
	auto system = smart_refctd_ptr<ISystem>(params.assetManager->getSystem());
	auto* set = params.assetManager->getCompilerSet();
	auto compiler = set->getShaderCompiler(IShader::E_CONTENT_TYPE::ECT_HLSL);
	auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(system));
	auto includeLoader = includeFinder->getDefaultFileSystemLoader();
	includeFinder->addSearchPath(NBL_ARCHIVE_ENTRY.data(), includeLoader);

	auto compileShader = [&](const std::string& filePath, IShader::E_SHADER_STAGE stage) -> smart_refctd_ptr<IShader>
		{
			IAssetLoader::SAssetLoadParams lparams = {};
			lparams.logger = params.utilities->getLogger();
			lparams.workingDirectory = NBL_ARCHIVE_ENTRY.data();
			auto bundle = params.assetManager->getAsset(filePath, lparams);
			if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
			{
				params.utilities->getLogger()->log("Shader %s not found!", ILogger::ELL_ERROR, filePath.c_str());
				exit(-1);
			}

			const auto assets = bundle.getContents();
			assert(assets.size() == 1);
			smart_refctd_ptr<IShader> shaderSrc = IAsset::castDown<IShader>(assets[0]);
			if (!shaderSrc)
				return nullptr;

			CHLSLCompiler::SOptions options = {};
			options.stage = stage;
			options.preprocessorOptions.sourceIdentifier = filePath;
			options.preprocessorOptions.logger = params.utilities->getLogger();
			options.preprocessorOptions.includeFinder = includeFinder.get();
			shaderSrc = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);

			return params.utilities->getLogicalDevice()->compileShader({ shaderSrc.get() });
		};

	if (!system->exists(path(NBL_ARCHIVE_ENTRY) / "common.hlsl", {}))
		mount(smart_refctd_ptr<ILogger>(params.utilities->getLogger()), system.get(), NBL_ARCHIVE_ENTRY);

	auto vertexShader = compileShader(vsPath, IShader::E_SHADER_STAGE::ESS_VERTEX);
	auto fragmentShader = compileShader(fsPath, IShader::E_SHADER_STAGE::ESS_FRAGMENT);

	if (!vertexShader || !fragmentShader)
	{
		params.utilities->getLogger()->log("Could not compile shaders!", ILogger::ELL_ERROR);
		return nullptr;
	}

	video::IGPUGraphicsPipeline::SCreationParams pipelineParams[1] = {};
	pipelineParams[0].layout = pipelineLayout;
	pipelineParams[0].vertexShader = { .shader = vertexShader.get(), .entryPoint = "main" };
	pipelineParams[0].fragmentShader = { .shader = fragmentShader.get(), .entryPoint = "main" };
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

			if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
				flags |= IDeviceMemoryAllocation::EMCAF_READ;
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
		std::make_pair(bool(buffer->getMemoryReqs().memoryTypeBits & params.utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits()), "Streaming buffer must have up-streaming memory type bits enabled!"),
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

	IGPUBuffer::SCreationParams bufparams;
	bufparams.size = sizeof(uint32_t) * unitAABBIndices.size();
	bufparams.usage = IGPUBuffer::EUF_INDEX_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;

	smart_refctd_ptr<IGPUBuffer> indicesBuffer;
	params.utilities->createFilledDeviceLocalBufferOnDedMem(
		SIntendedSubmitInfo{ .queue = params.transfer },
		std::move(bufparams),
		unitAABBIndices.data()
	).move_into(indicesBuffer);

	return indicesBuffer;
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> DrawAABB::createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange)
{
	return device->createPipelineLayout({ &pcRange , 1 }, nullptr, nullptr, nullptr, nullptr);
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> DrawAABB::createDefaultPipelineLayout(video::ILogicalDevice* device)
{
	SPushConstantRange pcRange = {
		.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX,
		.offset = 0,
		.size = sizeof(SPushConstants)
	};
	return device->createPipelineLayout({ &pcRange , 1 }, nullptr, nullptr, nullptr, nullptr);
}

bool DrawAABB::renderSingle(IGPUCommandBuffer* commandBuffer, const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t4& color, const hlsl::float32_t4x4& cameraMat)
{
	if (!(m_cachedCreationParams.drawMode & ADM_DRAW_SINGLE))
	{
		m_cachedCreationParams.utilities->getLogger()->log("DrawAABB has not been enabled for draw single!", ILogger::ELL_ERROR);
		return false;
	}

	commandBuffer->bindGraphicsPipeline(m_singlePipeline.get());
	commandBuffer->setLineWidth(1.f);
	asset::SBufferBinding<video::IGPUBuffer> indexBinding = { .offset = 0, .buffer = m_indicesBuffer };
	commandBuffer->bindIndexBuffer(indexBinding, asset::EIT_32BIT);

	SSinglePushConstants pc;

	hlsl::float32_t4x4 instanceTransform = getTransformFromAABB(aabb);
	pc.instance.transform = hlsl::mul(cameraMat, instanceTransform);
	pc.instance.color = color;
	
	commandBuffer->pushConstants(m_singlePipeline->getLayout(), ESS_VERTEX, 0, sizeof(SSinglePushConstants), &pc);
	commandBuffer->drawIndexed(IndicesCount, 1, 0, 0, 0);

	return true;
}

//bool DrawAABB::render(IGPUCommandBuffer* commandBuffer, ISemaphore::SWaitInfo waitInfo, std::span<const InstanceData> aabbInstances, const hlsl::float32_t4x4& cameraMat)
//{
//	
//}

hlsl::float32_t4x4 DrawAABB::getTransformFromAABB(const hlsl::shapes::AABB<3, float>& aabb)
{
	const auto diagonal = aabb.getExtent();
	hlsl::float32_t4x4 transform;
	transform[0][3] = aabb.minVx.x;
	transform[1][3] = aabb.minVx.y;
	transform[2][3] = aabb.minVx.z;
	transform[3][3] = 1.f;
	transform[0][0] = diagonal.x;
	transform[1][1] = diagonal.y;
	transform[2][2] = diagonal.z;
	return transform;
}

}
