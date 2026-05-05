// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/Frustum/CDrawFrustum.h"

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/ext/frustum/builtin/build/CArchive.h"
#endif

#include "nbl/ext/Frustum/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;
using namespace hlsl;

namespace nbl::ext::frustum
{

	core::smart_refctd_ptr<CDrawFrustum> CDrawFrustum::create(SCreationParameters&& params)
	{
		auto* const logger = params.utilities->getLogger();

		if (!params.validate())
		{
			logger->log("Failed creation parameters validation!", ILogger::ELL_ERROR);
			return nullptr;
		}

		ConstructorParams constructorParams;

		if (params.drawMode & DM_SINGLE)
		{
			auto pipelineLayout = params.singlePipelineLayout;
			if (!pipelineLayout)
				pipelineLayout = createDefaultPipelineLayout(params.utilities->getLogicalDevice(), DM_SINGLE);
			constructorParams.singlePipeline = createPipeline(params, pipelineLayout.get(), DM_SINGLE);
			if (!constructorParams.singlePipeline)
			{
				logger->log("Failed to create pipeline!", ILogger::ELL_ERROR);
				return nullptr;
			}
		}

		if (params.drawMode & DM_BATCH)
		{
			auto pipelineLayout = params.batchPipelineLayout;
			if (!pipelineLayout)
				pipelineLayout = createDefaultPipelineLayout(params.utilities->getLogicalDevice(), DM_BATCH);
			constructorParams.batchPipeline = createPipeline(params, pipelineLayout.get(), DM_BATCH);
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
		return core::smart_refctd_ptr<CDrawFrustum>(new CDrawFrustum(std::move(constructorParams)));
	}

	constexpr std::string_view NBL_EXT_MOUNT_ENTRY = "nbl/ext/Frustum";

	const smart_refctd_ptr<IFileArchive> CDrawFrustum::mount(smart_refctd_ptr<ILogger> logger, ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias)
	{
		assert(system);

		if (!system)
			return nullptr;

		const auto composed = path(archiveAlias.data()) / nbl::ext::frustum::builtin::build::get_spirv_key<"draw_frustum">(device);
		if (system->exists(composed, {}))
			return nullptr;

#ifdef NBL_EMBED_BUILTIN_RESOURCES
		auto archive = make_smart_refctd_ptr<builtin::build::CArchive>(smart_refctd_ptr(logger));
#else
		auto archive = make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(std::string_view(NBL_FRUSTUM_HLSL_MOUNT_POINT), smart_refctd_ptr(logger), system);
#endif

		system->mount(smart_refctd_ptr(archive), archiveAlias.data());
		return smart_refctd_ptr(archive);
	}

	smart_refctd_ptr<IGPUGraphicsPipeline> CDrawFrustum::createPipeline(SCreationParameters& params, const IGPUPipelineLayout* pipelineLayout, DrawMode mode)
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

		const auto key = nbl::ext::frustum::builtin::build::get_spirv_key<"draw_frustum">(device);
		smart_refctd_ptr<IShader> unifiedShader = getShader(key);
		if (!unifiedShader)
		{
			params.utilities->getLogger()->log("Could not compile shaders!", ILogger::ELL_ERROR);
			return nullptr;
		}

		video::IGPUGraphicsPipeline::SCreationParams pipelineParams[1] = {};
		pipelineParams[0].layout = pipelineLayout;
		pipelineParams[0].vertexShader = { .shader = unifiedShader.get(), .entryPoint = (mode & DM_SINGLE) ? "frustum_vertex_single" : "frustum_vertex_instances" };
		pipelineParams[0].fragmentShader = { .shader = unifiedShader.get(), .entryPoint = "frustum_fragment" };
		asset::SRasterizationParams rasterParams;
		rasterParams.depthWriteEnable = true;
		rasterParams.depthCompareOp = asset::ECO_GREATER;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		pipelineParams[0].cached = {
			.primitiveAssembly = {
				.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_LINE_LIST,
			},
			.rasterization = rasterParams,
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

	bool CDrawFrustum::createStreamingBuffer(SCreationParameters& params)
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
			buffer->setObjectDebugName("Frustum Streaming Buffer");

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
			std::make_pair(binding.memory->isCurrentlyMapped(), "Streaming buffer's memory must be mapped!"), 
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

	smart_refctd_ptr<IGPUBuffer> CDrawFrustum::createIndicesBuffer(SCreationParameters& params)
	{
		std::array<uint32_t, IndicesCount> unitFrustumIndices;
		unitFrustumIndices[0] = 0;
		unitFrustumIndices[1] = 1;
		unitFrustumIndices[2] = 0;
		unitFrustumIndices[3] = 2;

		unitFrustumIndices[4] = 3;
		unitFrustumIndices[5] = 1;
		unitFrustumIndices[6] = 3;
		unitFrustumIndices[7] = 2;

		unitFrustumIndices[8] = 4;
		unitFrustumIndices[9] = 5;
		unitFrustumIndices[10] = 4;
		unitFrustumIndices[11] = 6;

		unitFrustumIndices[12] = 7;
		unitFrustumIndices[13] = 5;
		unitFrustumIndices[14] = 7;
		unitFrustumIndices[15] = 6;

		unitFrustumIndices[16] = 0;
		unitFrustumIndices[17] = 4;
		unitFrustumIndices[18] = 1;
		unitFrustumIndices[19] = 5;

		unitFrustumIndices[20] = 2;
		unitFrustumIndices[21] = 6;
		unitFrustumIndices[22] = 3;
		unitFrustumIndices[23] = 7;

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
		bufparams.size = sizeof(uint32_t) * unitFrustumIndices.size();
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
			cmdbuf->updateBuffer(bufRange, unitFrustumIndices.data());

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

	core::smart_refctd_ptr<video::IGPUPipelineLayout> CDrawFrustum::createPipelineLayoutFromPCRange(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange)
	{
		return device->createPipelineLayout({ &pcRange , 1 }, nullptr, nullptr, nullptr, nullptr);
	}

	core::smart_refctd_ptr<video::IGPUPipelineLayout> CDrawFrustum::createDefaultPipelineLayout(video::ILogicalDevice* device, DrawMode mode)
	{
		const uint32_t offset = (mode & DM_BATCH) ? offsetof(ext::frustum::PushConstants, ipc) : offsetof(ext::frustum::PushConstants, spc);
		const uint32_t pcSize = (mode & DM_BATCH) ? sizeof(SInstancedPC) : sizeof(SSinglePC);
		SPushConstantRange pcRange = {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX,
			.offset = offset,
			.size = pcSize
		};
		return createPipelineLayoutFromPCRange(device, pcRange);
	}

	bool CDrawFrustum::renderSingle(const DrawParameters& params, const hlsl::float32_t4x4& frustumTransform, const hlsl::float32_t4& color)
	{
		if (!(m_cachedCreationParams.drawMode & DM_SINGLE))
		{
			m_cachedCreationParams.utilities->getLogger()->log("CDrawFrustum has not been enabled for draw single!", ILogger::ELL_ERROR);
			return false;
		}

		auto& commandBuffer = params.commandBuffer;
		commandBuffer->bindGraphicsPipeline(m_singlePipeline.get());
		commandBuffer->setLineWidth(params.lineWidth);
		asset::SBufferBinding<video::IGPUBuffer> indexBinding = { .offset = 0, .buffer = m_indicesBuffer };
		commandBuffer->bindIndexBuffer(indexBinding, asset::EIT_32BIT);

		SSinglePC pc;
		pc.instance.transform = hlsl::mul(params.viewProjectionMatrix, frustumTransform);
		pc.instance.color = color;

		commandBuffer->pushConstants(m_singlePipeline->getLayout(), ESS_VERTEX, offsetof(ext::frustum::PushConstants, spc), sizeof(SSinglePC), &pc);
		commandBuffer->drawIndexed(IndicesCount, 1, 0, 0, 0);

		return true;
	}

	bool CDrawFrustum::render(const DrawParameters& params, video::ISemaphore::SWaitInfo waitInfo, std::span<const InstanceData> frustumInstances)
	{
		system::logger_opt_ptr logger = m_cachedCreationParams.utilities->getLogger();
		if (!(m_cachedCreationParams.drawMode & DM_BATCH))
		{
			logger.log("CDrawFrustum has not been enabled for draw batches!", system::ILogger::ELL_ERROR);
			return false;
		}

		using offset_t = SCachedCreationParameters::streaming_buffer_t::size_type;
		constexpr offset_t MaxAlignment = sizeof(InstanceData);
		const auto MaxPOTAlignment = hlsl::roundUpToPoT(MaxAlignment);
		auto* streaming = m_cachedCreationParams.streamingBuffer.get();
		if (streaming->getAddressAllocator().max_alignment() < MaxPOTAlignment)
		{
			logger.log("Draw Frustum Streaming Buffer cannot guarantee the alignments we require!");
			return false;
		}

		auto* const streamingPtr = reinterpret_cast<uint8_t*>(streaming->getBufferPointer());
		assert(streamingPtr);

		auto& commandBuffer = params.commandBuffer;
		commandBuffer->bindGraphicsPipeline(m_batchPipeline.get());
		commandBuffer->setLineWidth(params.lineWidth);
		asset::SBufferBinding<video::IGPUBuffer> indexBinding = { .offset = 0, .buffer = m_indicesBuffer };
		commandBuffer->bindIndexBuffer(indexBinding, asset::EIT_32BIT);

		auto srcIt = frustumInstances.begin();
		auto setInstancesRange = [&](InstanceData* data, uint32_t count) -> void {
			for (uint32_t i = 0; i < count; i++)
			{
				auto inst = data + i;
				*inst = *srcIt;
				inst->transform = hlsl::mul(params.viewProjectionMatrix, inst->transform);
				srcIt++;

				if (srcIt == frustumInstances.end())
					break;
			}
		};

		const uint32_t numInstances = frustumInstances.size();
		uint32_t remainingInstancesBytes = numInstances * sizeof(InstanceData);
		while (srcIt != frustumInstances.end())
		{
			uint32_t blockByteSize = core::alignUp(remainingInstancesBytes, MaxAlignment);
			bool allocated = false;

			offset_t blockOffset = SCachedCreationParameters::streaming_buffer_t::invalid_value;
			const uint32_t smallestAlloc = hlsl::max<uint32_t>(core::alignUp(sizeof(InstanceData), MaxAlignment), streaming->getAddressAllocator().min_size());
			while (blockByteSize >= smallestAlloc)
			{
				std::chrono::steady_clock::time_point waitTill = std::chrono::steady_clock::now() + std::chrono::milliseconds(1u);
				if (streaming->multi_allocate(waitTill, 1, &blockOffset, &blockByteSize, &MaxAlignment) == 0u)
				{
					allocated = true;
					break;
				}

				streaming->cull_frees();
				blockByteSize >>= 1;
			}

			if (!allocated)
			{
				logger.log("Failed to allocate a chunk from streaming buffer for the next drawcall batch.", system::ILogger::ELL_ERROR);
				return false;
			}

			const uint32_t instanceCount = blockByteSize / sizeof(InstanceData);
			auto* const streamingInstancesPtr = reinterpret_cast<InstanceData*>(streamingPtr + blockOffset);
			setInstancesRange(streamingInstancesPtr, instanceCount);

			if (streaming->needsManualFlushOrInvalidate())
			{
				const video::ILogicalDevice::MappedMemoryRange flushRange(streaming->getBuffer()->getBoundMemory().memory, blockOffset, blockByteSize);
				m_cachedCreationParams.utilities->getLogicalDevice()->flushMappedMemoryRanges(1, &flushRange);
			}

			remainingInstancesBytes -= instanceCount * sizeof(InstanceData);

			SInstancedPC pc;
			pc.pInstanceBuffer = m_cachedCreationParams.streamingBuffer->getBuffer()->getDeviceAddress() + blockOffset;

			commandBuffer->pushConstants(m_batchPipeline->getLayout(), asset::IShader::E_SHADER_STAGE::ESS_VERTEX, offsetof(ext::frustum::PushConstants, ipc), sizeof(SInstancedPC), &pc);
			commandBuffer->drawIndexed(IndicesCount, instanceCount, 0, 0, 0);

			streaming->multi_deallocate(1, &blockOffset, &blockByteSize, waitInfo);
		}

		return true;
	}

}
