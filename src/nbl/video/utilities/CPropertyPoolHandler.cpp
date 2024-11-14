#include "nbl/video/utilities/CPropertyPoolHandler.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace video;

#if 0 // TODO: port
//
CPropertyPoolHandler::CPropertyPoolHandler(core::smart_refctd_ptr<ILogicalDevice>&& device) : m_device(std::move(device)), m_dsCache()
{
	// TODO: rewrite in HLSL!
#if 0
	const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
	m_maxPropertiesPerPass = core::min<uint32_t>((deviceLimits.maxPerStageDescriptorSSBOs-2u)/2u,MaxPropertiesPerDispatch);
	m_alignment = core::max(deviceLimits.minSSBOAlignment,256u/*TODO: deviceLimits.nonCoherentAtomSize*/);

	auto system = m_device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<asset::ICPUBuffer> glsl;
	{
		auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
		{
			nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
			system->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
			if (future.wait())
				return future.copy();
			return nullptr;
		};

		auto glslFile = loadBuiltinData("nbl/builtin/glsl/property_pool/copy.comp");
		glsl = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
		memcpy(glsl->getPointer(), glslFile->getMappedPointer(), glsl->getSize());
	}

	auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, "????");
	auto gpushader = m_device->createShader(asset::CGLSLCompiler::createOverridenCopy(cpushader.get(), "\n#define NBL_BUILTIN_MAX_PROPERTIES_PER_PASS %d\n", m_maxPropertiesPerPass));
	auto specshader = m_device->createSpecializedShader(gpushader.get(), { nullptr,nullptr,"main"});

	const auto maxStreamingAllocations = 2u*m_maxPropertiesPerPass+2u;
	//m_tmpAddressRanges = reinterpret_cast<AddressUploadRange*>(malloc((sizeof(AddressUploadRange)+sizeof(uint32_t)*3u)*maxStreamingAllocations));
	
	IGPUDescriptorSetLayout::SBinding bindings[4];
	for (auto j=0; j<4; j++)
	{
		bindings[j].binding = j;
		bindings[j].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
		bindings[j].count = j<2u ? 1u:m_maxPropertiesPerPass;
		bindings[j].stageFlags = asset::IShader::ESS_COMPUTE;
		bindings[j].samplers = nullptr;
	}
	auto dsLayout = m_device->createDescriptorSetLayout(bindings,bindings+4);
	// TODO: if we decide to invalidate all cmdbuffs used for updates (make them non reusable), then we can use the ECF_NONE flag
	auto descPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT,&dsLayout.get(),&dsLayout.get()+1u,&CPropertyPoolHandler::DescriptorCacheSize);
	m_dsCache = core::make_smart_refctd_ptr<TransferDescriptorSetCache>(m_device.get(),std::move(descPool),core::smart_refctd_ptr(dsLayout));
	
	const asset::SPushConstantRange baseDWORD = {asset::IShader::ESS_COMPUTE,0u,sizeof(uint32_t)*2u};
	auto layout = m_device->createPipelineLayout(&baseDWORD,&baseDWORD+1u,std::move(dsLayout));
	m_pipeline = m_device->createComputePipeline(nullptr,std::move(layout),std::move(specshader));
#endif
}


bool CPropertyPoolHandler::transferProperties(
	IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence,
	const asset::SBufferBinding<video::IGPUBuffer>& scratch, const asset::SBufferBinding<video::IGPUBuffer>& addresses,
	const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd,
	system::logger_opt_ptr logger, const uint32_t baseDWORD, const uint32_t endDWORD
)
{
	assert(false); // TODO: Atil
#if 0
	if (requestsBegin==requestsEnd)
		return true;
	if (!scratch.buffer || !scratch.buffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF))
	{
		logger.log("CPropertyPoolHandler: Need a valid scratch buffer which can have updates staged from the commandbuffer!",system::ILogger::ELL_ERROR);
		return false;
	}
	// TODO: validate usage flags
	if (scratch.offset+getMaxScratchSize()>scratch.buffer->getSize())
		logger.log("CPropertyPoolHandler: The scratch buffer binding provided might not be big enough in the worst case!",system::ILogger::ELL_WARNING);

	const auto totalProps = std::distance(requestsBegin,requestsEnd);
	const auto fullPasses = totalProps/m_maxPropertiesPerPass;
				
	nbl_glsl_property_pool_transfer_t transferData[MaxPropertiesPerDispatch];
	// TODO: factor out this function to be directly used in the streaming transfer, also split out the validation and allow it to use a pre-acquired descriptor set
	auto copyPass = [&](const TransferRequest* localRequests, uint32_t propertiesThisPass) -> bool
	{
		const auto scratchSize = sizeof(nbl_glsl_property_pool_transfer_t)*propertiesThisPass;
		if (scratch.offset+scratchSize>scratch.buffer->getSize())
		{
			logger.log("CPropertyPoolHandler: The scratch buffer binding provided is not big enough!",system::ILogger::ELL_ERROR);
			return false;
		}
		// count max dwords to transfer and check element size divisible by `sizeof(uint)`
		uint32_t maxDWORDs = 0u;
		for (uint32_t i=0; i<propertiesThisPass; i++)
		{
			const auto& request = localRequests[i];
			if (request.elementSize%sizeof(uint32_t))
			{
				logger.log("CPropertyPoolHandler::TransferRequest::elementSize (was %d) must be aligned to 4 bytes!",system::ILogger::ELL_ERROR,request.elementSize);
				assert(false);
				return false;
			}
			const auto elementsByteSize = request.elementCount*request.elementSize;
			maxDWORDs = core::max<uint32_t>(elementsByteSize/sizeof(uint32_t),maxDWORDs);
		}
		maxDWORDs = core::min(maxDWORDs,endDWORD);
		if (maxDWORDs<=baseDWORD)
			return true;
		
		// update desc sets (TODO: handle acquire failure, by using push descriptors!)
		// TODO: acquire the set just once for all streaming chunked dispatches (scratch, addresses, and destinations dont change)
		// however this will require the usage of IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC for the source data bindings
		auto setIx = m_dsCache->acquireSet(this,scratch,addresses,localRequests,propertiesThisPass);
		if (setIx==IDescriptorSetCache::invalid_index)
		{
			logger.log("CPropertyPoolHandler: Failed to acquire descriptor set!",system::ILogger::ELL_ERROR);
			return false;
		}

		// prepare the transfers
		for (uint32_t i=0; i<propertiesThisPass; i++)
		{
			const auto& request = localRequests[i];

			auto& transfer = transferData[i];
			transfer.propertyDWORDsize_flags = request.elementSize/sizeof(uint32_t);
			transfer.propertyDWORDsize_flags |= uint32_t(request.flags)<<(32-TransferRequest::EF_BIT_COUNT);
			transfer.elementCount = request.elementCount;
			//
			transfer.srcIndexOffset = request.srcAddressesOffset;
			transfer.dstIndexOffset = request.dstAddressesOffset;
		}
		cmdbuf->updateBuffer(scratch.buffer.get(),scratch.offset,sizeof(nbl_glsl_property_pool_transfer_t)*propertiesThisPass,transferData);
		video::IGPUCommandBuffer::SBufferMemoryBarrier buffBarrier;
		{
			buffBarrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			buffBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			buffBarrier.dstQueueFamilyIndex = buffBarrier.srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
			buffBarrier.buffer = scratch.buffer;
			buffBarrier.offset = scratch.offset;
			buffBarrier.size = scratchSize;
			cmdbuf->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,asset::EDF_NONE,0u,nullptr,1u,&buffBarrier,0u,nullptr);
		}
		cmdbuf->bindComputePipeline(m_pipeline.get());
		// bind desc sets
		auto set = m_dsCache->getSet(setIx);
		cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_pipeline->getLayout(),0u,1u,&set);
		{
			const uint32_t data[] = {baseDWORD,endDWORD};
			cmdbuf->pushConstants(m_pipeline->getLayout(),asset::IShader::ESS_COMPUTE,0u,sizeof(data),data);
		}
		// dispatch
		{
			const auto& limits = m_device->getPhysicalDevice()->getLimits();
			const auto invocationCoarseness = limits.maxOptimallyResidentWorkgroupInvocations*propertiesThisPass;
			cmdbuf->dispatch(limits.computeOptimalPersistentWorkgroupDispatchSize(maxDWORDs-baseDWORD,invocationCoarseness),propertiesThisPass,1u);
		}
		{
			buffBarrier.barrier.srcAccessMask = asset::EAF_SHADER_READ_BIT;
			buffBarrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			cmdbuf->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,asset::EDF_NONE,0u,nullptr,1u,&buffBarrier,0u,nullptr);
		}
		// deferred release resources
		m_dsCache->releaseSet(m_device.get(),core::smart_refctd_ptr<IGPUFence>(fence),setIx);

		return true;
	};

	bool result = true;
	auto requests = requestsBegin;
	for (uint32_t i=0; i<fullPasses; i++)
	{
		result = copyPass(requests,m_maxPropertiesPerPass)&&result;
		requests += m_maxPropertiesPerPass;
	}

	const auto leftOverProps = totalProps-fullPasses*m_maxPropertiesPerPass;
	if (leftOverProps)
		result = copyPass(requests,leftOverProps)&&result;

	return result;
#endif
}

uint32_t CPropertyPoolHandler::transferProperties(
	StreamingTransientDataBufferMT<>* const upBuff, IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence, IQueue* const queue,
	const asset::SBufferBinding<video::IGPUBuffer>& scratch, UpStreamingRequest* &requests, const uint32_t requestCount,
	uint32_t& waitSemaphoreCount, IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite, const asset::PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore,
	system::logger_opt_ptr logger, const std::chrono::steady_clock::time_point& maxWaitPoint
)
{
	if (!requestCount)
		return 0u;

	// somewhat decent attempt at packing
	std::sort(requests,requests+requestCount,[](const UpStreamingRequest& rhs, const UpStreamingRequest& lhs)->bool{return rhs.getElementDWORDs()<lhs.getElementDWORDs();});
	
    class MemoryUsageIterator
    {
			const UpStreamingRequest* localRequests;
			uint32_t propertiesThisPass;
			uint32_t baseDWORDs;
			uint32_t dwordCount;
			uint32_t memoryConsumed;
		public:
			using value_type = uint32_t;
			using pointer = uint32_t*;
			using reference = uint32_t&;
			using difference_type = int64_t;
			using iterator_category = std::random_access_iterator_tag;

			MemoryUsageIterator(const UpStreamingRequest* _localRequests, const uint32_t _propertiesThisPass, const uint32_t _baseDWORDs, const uint32_t _dwordCount)
				: localRequests(_localRequests), propertiesThisPass(_propertiesThisPass), baseDWORDs(_baseDWORDs), dwordCount(_dwordCount), memoryConsumed(0u)
			{
				bool end = true;

				const auto cumulativeDWORDs = dwordCount+baseDWORDs;
				for (auto j=0; j<propertiesThisPass; j++)
				{
					const auto& request = localRequests[j];
					auto endDWORD = request.getElementDWORDs();
					if (cumulativeDWORDs<=endDWORD)
					{
						endDWORD = cumulativeDWORDs;
						end = false;
					}

					if (!request.source.device2device)
						memoryConsumed += (endDWORD-baseDWORDs)*sizeof(uint32_t);
				
					const auto indexConsumption = (request.getElementsToSkip(endDWORD+request.elementSize-1u)-request.getElementsToSkip(baseDWORDs))*sizeof(uint32_t);
					if (request.srcAddresses)
						memoryConsumed += request.fill ? sizeof(uint32_t):indexConsumption;
					if (request.dstAddresses)
						memoryConsumed += indexConsumption;
				}

				if (end)
					memoryConsumed = ~0u;
			}

			MemoryUsageIterator operator+(difference_type n) const
			{
				return MemoryUsageIterator(localRequests,propertiesThisPass,baseDWORDs,dwordCount+static_cast<uint32_t>(n));
			}
			MemoryUsageIterator& operator+=(difference_type n)
			{
				return operator=(operator+(n));
			}
			MemoryUsageIterator operator-(difference_type n) const
			{
				return operator+(-n);
			}
			MemoryUsageIterator& operator-=(difference_type n)
			{
				return operator+=(-n);
			}
			difference_type operator-(const MemoryUsageIterator& rhs) const
			{
				return static_cast<difference_type>(dwordCount-rhs.dwordCount);
			}
			MemoryUsageIterator& operator++()
			{
				return operator+=(1);
			}
			MemoryUsageIterator operator++(int)
			{
				auto cp = *this;
				++(*this);
				return cp;
			}
			MemoryUsageIterator& operator--()
			{
				return operator-=(1);
			}
			MemoryUsageIterator operator--(int)
			{
				auto cp = *this;
				--(*this);
				return cp;
			}

			value_type operator*() const
			{
				return memoryConsumed;
			}

			bool operator==(const MemoryUsageIterator& rhs) const
			{
				return dwordCount==rhs.dwordCount;
			}
			bool operator!=(const MemoryUsageIterator& rhs) const
			{
				return dwordCount!=rhs.dwordCount;
			}
			bool operator<(const MemoryUsageIterator& rhs) const
			{
				return this->operator-(rhs) > 0;
			}
			bool operator>(const MemoryUsageIterator& rhs) const
			{
				return rhs < (*this);
			}
			bool operator>=(const MemoryUsageIterator& rhs) const
			{
				return !(this->operator<(rhs));
			}
			bool operator<=(const MemoryUsageIterator& rhs) const
			{
				return !(this->operator>(rhs));
			}

			inline uint32_t getUsage() const
			{
				return memoryConsumed;
			}
	};

	//
	const auto& limits = m_device->getPhysicalDevice()->getLimits();

	//
	TransferRequest xfers[MaxPropertiesPerDispatch];
	uint8_t* const upBuffPtr8 = reinterpret_cast<uint8_t*>(upBuff->getBufferPointer());
	uint32_t* const upBuffPtr32 = reinterpret_cast<uint32_t*>(upBuff->getBufferPointer());
	const asset::SBufferBinding<video::IGPUBuffer> uploadBuffer = {0ull,core::smart_refctd_ptr<video::IGPUBuffer>(upBuff->getBuffer())};
	auto attempt = [&](const uint32_t baseDWORDs, const uint32_t remainingDWORDs, UpStreamingRequest* &localRequests, uint32_t& propertiesThisPass) -> uint32_t
	{
		assert(remainingDWORDs);
		// skip requests that won't participate
		while (propertiesThisPass && localRequests->getElementDWORDs()<=baseDWORDs)
		{
			localRequests++;
			propertiesThisPass--;
		}
		// nothing to do
		if (propertiesThisPass==0u)
			return 0u;
		
		const uint32_t worstCasePadding = m_alignment*propertiesThisPass-propertiesThisPass;
		// prevent micro dispatches
		const uint32_t minimumDWORDs = core::min(limits.maxResidentInvocations,remainingDWORDs);
		auto memoryUsage = MemoryUsageIterator(localRequests,propertiesThisPass,baseDWORDs,minimumDWORDs);

		uint32_t doneDWORDs = minimumDWORDs;
		// can we do better? lets check.
		{
			const uint32_t minimumMemoryNeeded = core::alignUp(memoryUsage.getUsage()+worstCasePadding,m_alignment);
			const uint32_t freeSpace = static_cast<uint32_t>(core::alignDown(upBuff->max_size(),m_alignment));

			if (freeSpace>minimumMemoryNeeded)
			{
				memoryUsage = std::upper_bound(
					memoryUsage,
					MemoryUsageIterator(localRequests,propertiesThisPass,baseDWORDs,remainingDWORDs+1u),
					freeSpace-worstCasePadding
				);
				memoryUsage--;
				doneDWORDs = std::distance(MemoryUsageIterator(localRequests,propertiesThisPass,baseDWORDs,0u),memoryUsage);
				assert(doneDWORDs>=minimumDWORDs);
			}
		}

		constexpr auto invalid_address = video::StreamingTransientDataBufferMT<>::invalid_value;
		auto addr = invalid_address;
		const auto size = static_cast<uint32_t>(core::alignUp(memoryUsage.getUsage()+worstCasePadding,m_alignment));
		// because right now the GPUEventWrapper cant distinguish between placeholder fences and fences which will actually be signalled
		auto waitUntil = std::min(video::GPUEventWrapper::default_wait(),maxWaitPoint);
		upBuff->multi_allocate(waitUntil,1u,&addr,&size,&m_alignment);
		if (addr!=invalid_address)
		{
			const auto endDWORD = baseDWORDs+doneDWORDs;

			uint32_t offset = addr;
			// source data
			for (auto i=0u; i<propertiesThisPass; i++)
			{
				const auto& request = localRequests[i];
				auto& transfer = xfers[i];
				transfer.memblock = request.destination;
				transfer.flags = request.fill ? TransferRequest::EF_FILL:TransferRequest::EF_NONE;
				transfer.elementSize = request.elementSize;
				transfer.elementCount = request.elementCount;

				if (request.source.device2device)
					transfer.buffer = request.source.buffer;
				else
				{
					transfer.buffer = uploadBuffer;
					transfer.buffer.offset = offset;
					// copy
					const auto sizeDWORDs = core::min(request.getElementDWORDs(),endDWORD)-baseDWORDs;
					const auto bytesize = sizeDWORDs*sizeof(uint32_t);
					memcpy(upBuffPtr8+offset,reinterpret_cast<const uint32_t*>(request.source.data)+baseDWORDs,bytesize);
					// advance
					offset = core::alignUp(offset+bytesize,m_alignment);
				}
			}
			// addresses
			offset /= sizeof(uint32_t);
			for (auto i=0u; i<propertiesThisPass; i++)
			{
				const auto& request = localRequests[i];
				auto& transfer = xfers[i];
				
				const auto firstIndex = request.getElementsToSkip(baseDWORDs);
				const auto indexCount = request.getElementsToSkip(endDWORD+request.elementSize-1u)-firstIndex;
				const auto indexConsumption = indexCount*sizeof(uint32_t);
				if (request.srcAddresses)
				{
					transfer.srcAddressesOffset = offset;
					if (request.fill)
					{
						memcpy(upBuffPtr32+offset,request.srcAddresses,sizeof(uint32_t));
						++offset;
					}
					else
					{
						memcpy(upBuffPtr32+offset,request.srcAddresses+firstIndex,indexConsumption);
						offset += indexCount;
					}
				}
				else
					transfer.srcAddressesOffset = IPropertyPool::invalid;
				if (request.dstAddresses)
				{
					memcpy(upBuffPtr32+offset,request.dstAddresses+firstIndex,indexConsumption);
					transfer.dstAddressesOffset = offset;
					offset += indexCount;
				}
				else
					transfer.dstAddressesOffset = IPropertyPool::invalid;
			}
			assert(offset*sizeof(uint32_t)<=size+addr);
			// flush if needed
			if (upBuff->needsManualFlushOrInvalidate())
			{
				IDeviceMemoryAllocation::MappedMemoryRange flushRange;
				flushRange.memory = uploadBuffer.buffer->getBoundMemory();
				flushRange.range = {addr,size};
				m_device->flushMappedMemoryRanges(1u,&flushRange);
			}
			// no pipeline barriers necessary because write and optional flush happens before submit, and memory allocation is reclaimed after fence signal
			if (transferProperties(cmdbuf,fence,scratch,uploadBuffer,xfers,xfers+propertiesThisPass,logger,baseDWORDs,endDWORD))
			{
				upBuff->multi_deallocate(1u,&addr,&size,core::smart_refctd_ptr<IGPUFence>(fence),&cmdbuf);
				return doneDWORDs;
			}
			upBuff->multi_deallocate(1u,&addr,&size);
		}
		return 0u;
	};
	auto submit = [&]() -> void
	{
		cmdbuf->end();
		IQueue::SSubmitInfo submit;
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cmdbuf;
		submit.signalSemaphoreCount = 0u;
		submit.pSignalSemaphores = nullptr;
		assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
		submit.waitSemaphoreCount = waitSemaphoreCount;
		submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
		submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
		queue->submit(1u,&submit,fence);
		m_device->blockForFences(1u,&fence);
		waitSemaphoreCount = 0u;
		semaphoresToWaitBeforeOverwrite = nullptr;
		stagesToWaitForPerSemaphore = nullptr;
		// before resetting we need poll all events in the allocator's deferred free list
		upBuff->cull_frees();
		m_dsCache->poll_all();
		// we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
		m_device->resetFences(1u,&fence);
		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
	};
	// return remaining DWORDs
	auto copyPass = [&](UpStreamingRequest* &localRequests, uint32_t propertiesThisPass) -> uint32_t
	{
		const auto localRequestsEnd = localRequests+propertiesThisPass;
		// figure out how much work we have to do
		uint32_t maxDWORDs = 0u;
		for (uint32_t i=0; i<propertiesThisPass; i++)
		{
			const auto& request = localRequests[i];
			maxDWORDs = core::max<uint32_t>(request.getElementDWORDs(),maxDWORDs);
		}
		if (maxDWORDs==0u)
			return 0u;
		// TODO: acquire and update a descriptor set up front for all chunks (much faster than reacquire and update for every tiny chunk that transfers from same source to same destination)
		// do the transfers
		uint32_t doneDWORDs=0u;
		for (uint32_t submitDWORDs=~0u; doneDWORDs<maxDWORDs;)
		{
			const auto thisDWORDs = attempt(doneDWORDs,maxDWORDs-doneDWORDs,localRequests,propertiesThisPass);
			if (thisDWORDs==0u)
			{
				if (submitDWORDs)
				{
					submit();
					submitDWORDs = 0u;
					continue; // try again
				}
				else // second fail in a row
					break;
			}
			submitDWORDs += thisDWORDs;
			doneDWORDs += thisDWORDs;
		}
		if (doneDWORDs!=maxDWORDs)
			return maxDWORDs-doneDWORDs;
		requests = localRequestsEnd;
		return 0u;
	};

	const auto fullPasses = requestCount/m_maxPropertiesPerPass;
	// transfer as many properties at once as possible
	for (uint32_t i=0; i<fullPasses; i++)
	{
		const auto remainingDWORDs = copyPass(requests,m_maxPropertiesPerPass);
		if (remainingDWORDs)
			return remainingDWORDs;
	}

	const auto leftOverProps = requestCount-fullPasses*m_maxPropertiesPerPass;
	if (leftOverProps)
		return copyPass(requests,leftOverProps);
	
	return 0u;
}

uint32_t CPropertyPoolHandler::TransferDescriptorSetCache::acquireSet(
	CPropertyPoolHandler* handler, const asset::SBufferBinding<video::IGPUBuffer>& scratch, const asset::SBufferBinding<video::IGPUBuffer>& addresses,
	const TransferRequest* requests, const uint32_t propertyCount
)
{
	auto retval = IDescriptorSetCache::acquireSet();
	if (retval==IDescriptorSetCache::invalid_index)
		return IDescriptorSetCache::invalid_index;
	

	auto device = handler->getDevice();
	const auto maxPropertiesPerPass = handler->getMaxPropertiesPerTransferDispatch();


	IGPUDescriptorSet::SDescriptorInfo infos[MaxPropertiesPerDispatch*2u+2u];
	infos[0] = scratch;
	infos[0].info.buffer.size = sizeof(nbl_glsl_property_pool_transfer_t)*propertyCount;
	infos[1] = addresses;
	auto* inDescInfo = infos+2;
	auto* outDescInfo = infos+2+maxPropertiesPerPass;
	for (uint32_t i=0u; i<propertyCount; i++)
	{
		const auto& request = requests[i];
			
		const auto& memblock = request.memblock;

		// no not attempt to bind sized ranges of the buffers, remember that the copies are indexed, so the reads and writes may be scattered
		if (request.isDownload())
		{
			inDescInfo[i] = memblock;
			outDescInfo[i] = request.buffer;
		}
		else
		{
			inDescInfo[i] = request.buffer;
			outDescInfo[i] = memblock;
		}
	}
	// just to make Vulkan shut up
	for (uint32_t i=propertyCount; i<maxPropertiesPerPass; i++)
	{
		inDescInfo[i] = scratch;
		outDescInfo[i] = scratch;
	}
	IGPUDescriptorSet::SWriteDescriptorSet writes[4u];
	IGPUDescriptorSet* const set = IDescriptorSetCache::getSet(retval);
	for (auto i=0u; i<4u; i++)
	{
		writes[i].dstSet = set;
		writes[i].binding = i;
		writes[i].arrayElement = 0u;
		writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
	}
	writes[0].count = 1u;
	writes[0].info = infos;
	writes[1].count = 1u;
	writes[1].info = infos+1u;
	writes[2].count = maxPropertiesPerPass;
	writes[2].info = inDescInfo;
	writes[3].count = maxPropertiesPerPass;
	writes[3].info = outDescInfo;
	device->updateDescriptorSets(4u, writes, 0u, nullptr);

	return retval;
}
#endif