// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/DebugDraw/CDrawAABB.h"

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;
using namespace hlsl;

namespace nbl::ext::debugdraw
{

core::smart_refctd_ptr<DrawAABB> DrawAABB::create(SCreationParameters&& params)
{
	auto* const logger = params.utilities->getLogger();

	auto pipeline = createPipeline(params);
	if (!pipeline)
	{
		logger->log("Failed to create pipeline!", ILogger::ELL_ERROR);
		return nullptr;
	}

	if (!createStreamingBuffer(params))
	{
		logger->log("Failed to create streaming buffer!", ILogger::ELL_ERROR);
		return nullptr;
	}

    return core::smart_refctd_ptr<DrawAABB>(new DrawAABB(std::move(params), pipeline));
}

DrawAABB::DrawAABB(SCreationParameters&& params, smart_refctd_ptr<IGPUGraphicsPipeline> pipeline)
    : m_cachedCreationParams(std::move(params)), m_pipeline(pipeline)
{
	const auto unitAABB = core::aabbox3d<float>({ 0, 0, 0 }, { 1, 1, 1 });
	m_unitAABBVertices = getVerticesFromAABB(unitAABB);
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

smart_refctd_ptr<IGPUGraphicsPipeline> DrawAABB::createPipeline(SCreationParameters& params)
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

	if (!system->isDirectory(path(NBL_ARCHIVE_ENTRY.data())))
		mount(smart_refctd_ptr<ILogger>(params.utilities->getLogger()), system.get(), NBL_ARCHIVE_ENTRY);

	auto vertexShader = compileShader("aabb_instances.vertex.hlsl", IShader::E_SHADER_STAGE::ESS_VERTEX);
	auto fragmentShader = compileShader("aabb_instances.fragment.hlsl", IShader::E_SHADER_STAGE::ESS_FRAGMENT);

	video::IGPUGraphicsPipeline::SCreationParams pipelineParams[1] = {};
	pipelineParams[0].layout = params.pipelineLayout.get();
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

core::smart_refctd_ptr<video::IGPUPipelineLayout> DrawAABB::createDefaultPipelineLayout(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange)
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

smart_refctd_ptr<IGPUGraphicsPipeline> DrawAABB::createDefaultPipeline(video::ILogicalDevice* device, video::IGPUPipelineLayout* layout, video::IGPURenderpass* renderpass, video::IGPUGraphicsPipeline::SShaderSpecInfo& vertex, video::IGPUGraphicsPipeline::SShaderSpecInfo& fragment)
{
	smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;

	video::IGPUGraphicsPipeline::SCreationParams params[1] = {};
	params[0].layout = layout;
	params[0].vertexShader = vertex;
	params[0].fragmentShader = fragment;
	params[0].cached = {
		.primitiveAssembly = {
			.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_LINE_LIST,
		}
	};
	params[0].renderpass = renderpass;

	device->createGraphicsPipelines(nullptr, params, &pipeline);

	return pipeline;
}

bool DrawAABB::renderSingle(IGPUCommandBuffer* commandBuffer)
{
	commandBuffer->setLineWidth(1.f);
	commandBuffer->draw(24, 1, 0, 0);

	return true;
}

bool DrawAABB::render(IGPUCommandBuffer* commandBuffer, ISemaphore::SWaitInfo waitInfo, float* cameraMat3x4)
{
	using offset_t = SCachedCreationParameters::streaming_buffer_t::size_type;
	constexpr auto MdiSizes = std::to_array<offset_t>({ sizeof(float32_t3), sizeof(InstanceData) });
	// shared nPoT alignment needs to be divisible by all smaller ones to satisfy an allocation from all
	constexpr offset_t MaxAlignment = std::reduce(MdiSizes.begin(), MdiSizes.end(), 1, [](const offset_t a, const offset_t b)->offset_t {return std::lcm(a, b); });
	// allocator initialization needs us to round up to PoT
	const auto MaxPOTAlignment = roundUpToPoT(MaxAlignment);

	auto* streaming = m_cachedCreationParams.streamingBuffer.get();

	auto* const streamingPtr = reinterpret_cast<uint8_t*>(streaming->getBufferPointer());
	assert(streamingPtr);

	commandBuffer->bindGraphicsPipeline(m_pipeline.get());	// move outside of loop, only bind once
	commandBuffer->setLineWidth(1.f);

	auto instancesIt = m_instances.begin();
	const uint32_t verticesByteSize = sizeof(float32_t3) * m_unitAABBVertices.size();
	const uint32_t availableInstancesByteSize = streaming->getBuffer()->getSize() - verticesByteSize;
	const uint32_t instancesPerIter = availableInstancesByteSize / sizeof(InstanceData);
	using suballocator_t = core::LinearAddressAllocatorST<offset_t>;
	while (instancesIt != m_instances.end())
    {
		const uint32_t instanceCount = min(instancesPerIter, m_instances.size());
        offset_t inputOffset = 0u;
	    offset_t ImaginarySizeUpperBound = 0x1 << 30;
	    suballocator_t imaginaryChunk(nullptr, inputOffset, 0, roundUpToPoT(MaxAlignment), ImaginarySizeUpperBound);
	    uint32_t vertexByteOffset = imaginaryChunk.alloc_addr(verticesByteSize, sizeof(float32_t3));
	    uint32_t instancesByteOffset = imaginaryChunk.alloc_addr(sizeof(InstanceData) * instanceCount, sizeof(InstanceData));
	    const uint32_t totalSize = imaginaryChunk.get_allocated_size();

	    inputOffset = SCachedCreationParameters::streaming_buffer_t::invalid_value;
		std::chrono::steady_clock::time_point waitTill = std::chrono::steady_clock::now() + std::chrono::milliseconds(1u);
	    streaming->multi_allocate(waitTill, 1, &inputOffset, &totalSize, &MaxAlignment);

	    memcpy(streamingPtr + vertexByteOffset, m_unitAABBVertices.data(), sizeof(m_unitAABBVertices[0]) * m_unitAABBVertices.size());
	    memcpy(streamingPtr + instancesByteOffset, std::addressof(*instancesIt), sizeof(InstanceData) * instanceCount);
		instancesIt += instanceCount;

	    assert(!streaming->needsManualFlushOrInvalidate());

	    SPushConstants pc;
	    memcpy(pc.MVP, cameraMat3x4, sizeof(pc.MVP));
	    pc.pVertexBuffer = m_cachedCreationParams.streamingBuffer->getBuffer()->getDeviceAddress() + vertexByteOffset;
	    pc.pInstanceBuffer = m_cachedCreationParams.streamingBuffer->getBuffer()->getDeviceAddress() + instancesByteOffset;

	    commandBuffer->pushConstants(m_pipeline->getLayout(), ESS_VERTEX, 0, sizeof(SPushConstants), &pc);
	    commandBuffer->draw(m_unitAABBVertices.size(), instanceCount, 0, 0);

	    streaming->multi_deallocate(1, &inputOffset, &totalSize, waitInfo);
    }

	return true;
}

std::array<float32_t3, 24> DrawAABB::getVerticesFromAABB(const core::aabbox3d<float>& aabb)
{
	const auto& pMin = aabb.MinEdge;
	const auto& pMax = aabb.MaxEdge;

	std::array<float32_t3, 24> vertices;
	vertices[0] = float32_t3(pMin.X, pMin.Y, pMin.Z);
	vertices[1] = float32_t3(pMax.X, pMin.Y, pMin.Z);
	vertices[2] = float32_t3(pMin.X, pMin.Y, pMin.Z);
	vertices[3] = float32_t3(pMin.X, pMin.Y, pMax.Z);

	vertices[4] = float32_t3(pMax.X, pMin.Y, pMax.Z);
	vertices[5] = float32_t3(pMax.X, pMin.Y, pMin.Z);
	vertices[6] = float32_t3(pMax.X, pMin.Y, pMax.Z);
	vertices[7] = float32_t3(pMin.X, pMin.Y, pMax.Z);

	vertices[8] = float32_t3(pMin.X, pMax.Y, pMin.Z);
	vertices[9] = float32_t3(pMax.X, pMax.Y, pMin.Z);
	vertices[10] = float32_t3(pMin.X, pMax.Y, pMin.Z);
	vertices[11] = float32_t3(pMin.X, pMax.Y, pMax.Z);

	vertices[12] = float32_t3(pMax.X, pMax.Y, pMax.Z);
	vertices[13] = float32_t3(pMax.X, pMax.Y, pMin.Z);
	vertices[14] = float32_t3(pMax.X, pMax.Y, pMax.Z);
	vertices[15] = float32_t3(pMin.X, pMax.Y, pMax.Z);

	vertices[16] = float32_t3(pMin.X, pMin.Y, pMin.Z);
	vertices[17] = float32_t3(pMin.X, pMax.Y, pMin.Z);
	vertices[18] = float32_t3(pMax.X, pMin.Y, pMin.Z);
	vertices[19] = float32_t3(pMax.X, pMax.Y, pMin.Z);

	vertices[20] = float32_t3(pMin.X, pMin.Y, pMax.Z);
	vertices[21] = float32_t3(pMin.X, pMax.Y, pMax.Z);
	vertices[22] = float32_t3(pMax.X, pMin.Y, pMax.Z);
	vertices[23] = float32_t3(pMax.X, pMax.Y, pMax.Z);

	return vertices;
}

void DrawAABB::addAABB(const core::aabbox3d<float>& aabb, const hlsl::float32_t4& color)
{
	addAABB(shapes::AABB<3, float>{{aabb.MinEdge.X, aabb.MinEdge.Y, aabb.MinEdge.Z}, { aabb.MaxEdge.X, aabb.MaxEdge.Y, aabb.MaxEdge.Z }}, color);
}

void DrawAABB::addAABB(const hlsl::shapes::AABB<3,float>& aabb, const hlsl::float32_t4& color)
{
	const auto transform = hlsl::float32_t3x4(1);
	addOBB(aabb, transform, color);
}

void DrawAABB::addOBB(const hlsl::shapes::AABB<3, float>& aabb, const hlsl::float32_t3x4 transform, const hlsl::float32_t4& color)
{
	InstanceData instance;
	instance.color = color;

	core::matrix3x4SIMD instanceTransform;
	instanceTransform.setTranslation(core::vectorSIMDf(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, 0));
	const auto diagonal = aabb.getExtent();
	instanceTransform.setScale(core::vectorSIMDf(diagonal.x, diagonal.y, diagonal.z));

	core::matrix3x4SIMD worldTransform;
	memcpy(worldTransform.pointer(), &transform, sizeof(transform));

	instanceTransform = core::concatenateBFollowedByA(worldTransform, instanceTransform);
	memcpy(instance.transform, instanceTransform.pointer(), sizeof(core::matrix3x4SIMD));

	m_instances.push_back(instance);
}

void DrawAABB::clearAABBs()
{
	m_instances.clear();
}

}
