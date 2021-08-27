#include "nbl/video/utilities/CPropertyPoolHandler.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

#include "nbl/builtin/glsl/property_pool/transfer.glsl"
static_assert(_NBL_BUILTIN_PROPERTY_POOL_TRANSFER_T_SIZE==sizeof(nbl_glsl_property_pool_transfer_t));

using namespace nbl;
using namespace video;

//
CPropertyPoolHandler::CPropertyPoolHandler(core::smart_refctd_ptr<ILogicalDevice>&& device) : m_device(std::move(device)), m_dsCache()
{
	auto system = m_device->getPhysicalDevice()->getSystem();
	auto glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/property_pool/copy.comp")>();
	auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(glsl), asset::ICPUShader::buffer_contains_glsl);
	
	const auto maxSSBO = core::min<uint32_t>(m_device->getPhysicalDevice()->getLimits().maxPerStageSSBOs,MaxPropertyTransfers);
	m_maxPropertiesPerPass = (maxSSBO-1u)/2u;
	
	const auto maxStreamingAllocations = 2u*m_maxPropertiesPerPass+1u;
	{
		m_tmpIndexRanges = reinterpret_cast<IndexUploadRange*>(malloc((sizeof(IndexUploadRange)+sizeof(nbl_glsl_property_pool_transfer_t))*maxStreamingAllocations));
		m_tmpAddresses = reinterpret_cast<uint32_t*>(m_tmpIndexRanges+maxStreamingAllocations);
		m_tmpSizes = reinterpret_cast<uint32_t*>(m_tmpAddresses+maxStreamingAllocations);
		m_alignments = reinterpret_cast<uint32_t*>(m_tmpSizes+maxStreamingAllocations);
		std::fill_n(m_alignments,maxStreamingAllocations,m_device->getPhysicalDevice()->getLimits().SSBOAlignment);
	}

	auto shader = m_device->createGPUShader(asset::IGLSLCompiler::createOverridenCopy(cpushader.get(),"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n#define _NBL_BUILTIN_MAX_PROPERTIES_PER_COPY_ %d\n",IdealWorkGroupSize,m_maxPropertiesPerPass));
	auto specshader = m_device->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});
	
	IGPUDescriptorSetLayout::SBinding bindings[3];
	for (auto j=0; j<3; j++)
	{
		bindings[j].binding = j;
		bindings[j].type = asset::EDT_STORAGE_BUFFER;
		bindings[j].count = j ? m_maxPropertiesPerPass:1u;
		bindings[j].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bindings[j].samplers = nullptr;
	}
	auto dsLayout = m_device->createGPUDescriptorSetLayout(bindings,bindings+3);
	// TODO: if we decide to invalidate all cmdbuffs used for updates (make them non reusable), then we can use the ECF_NONE flag
	auto descPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT,&dsLayout.get(),&dsLayout.get()+1u,&CPropertyPoolHandler::DescriptorCacheSize);
	m_dsCache = core::make_smart_refctd_ptr<TransferDescriptorSetCache>(m_device.get(),std::move(descPool),core::smart_refctd_ptr(dsLayout));
	// TODO: push constants
	auto layout = m_device->createGPUPipelineLayout(nullptr,nullptr,std::move(dsLayout));
	m_pipeline = m_device->createGPUComputePipeline(nullptr,std::move(layout),std::move(specshader));
}

CPropertyPoolHandler::transfer_result_t CPropertyPoolHandler::addProperties(
	StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
	IGPUFence* const fence, const AllocationRequest* const requestsBegin, const AllocationRequest* const requestsEnd, system::logger_opt_ptr logger,
	const std::chrono::high_resolution_clock::time_point maxWaitPoint
)
{
	bool success = true;

	uint32_t transferCount = 0u;
	for (auto it=requestsBegin; it!=requestsEnd; it++)
	{
		const bool allocSuccess = it->pool->allocateProperties(it->outIndices.begin(),it->outIndices.end());
		if (!allocSuccess)
			logger.log("CPropertyPoolHandler: Failed to allocate %d properties from pool %p, part of request %d!",system::ILogger::ELL_WARNING,it->outIndices.size(),it->pool,it-requestsBegin);
		success = allocSuccess&&success;
		transferCount += it->pool->getPropertyCount();
	}

	core::vector<TransferRequest> transferRequests(transferCount);
	auto oit = transferRequests.begin();
	for (auto it=requestsBegin; it!=requestsEnd; it++)
	for (auto i=0u; i<it->pool->getPropertyCount(); i++)
	{
		oit->pool = it->pool;
		oit->indices = { it->outIndices.begin(),it->outIndices.end() };
		oit->propertyID = i;
		oit->data = it->data[i];
		oit++;
	}
	transfer_result_t result = transferProperties(upBuff,downBuff,cmdbuf,fence,transferRequests.data(),transferRequests.data()+transferCount,logger,maxWaitPoint);
	result.transferSuccess = result.transferSuccess&&success;
	return result;
}

CPropertyPoolHandler::transfer_result_t CPropertyPoolHandler::transferProperties(
	StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
	IGPUFence* const fence, const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd, system::logger_opt_ptr logger,
	const std::chrono::high_resolution_clock::time_point maxWaitPoint
)
{
	const auto totalProps = std::distance(requestsBegin,requestsEnd);

	transfer_result_t result =
	{
		download_future_t(
			core::smart_refctd_ptr(m_device),
			core::smart_refctd_ptr<StreamingTransientDataBufferMT<>>(upBuff),
			core::smart_refctd_ptr<IGPUFence>(fence),
			totalProps
		),true
	};
	if (totalProps!=0u)
	{
		const auto fullPasses = totalProps/m_maxPropertiesPerPass;

		constexpr auto invalid_address = std::remove_reference_t<decltype(upBuff->getAllocator())>::invalid_address;
		uint8_t* upBuffPtr = reinterpret_cast<uint8_t*>(upBuff->getBufferPointer());
				
		auto copyPass = [&](const TransferRequest* localRequests, uint32_t propertiesThisPass) -> bool
		{
			uint32_t upAllocations = 1u;
			uint32_t downAllocations = 0u;
			for (uint32_t i=0u; i<propertiesThisPass; i++)
			{
				if (localRequests[i].isDownload())
					downAllocations++;
				else
					upAllocations++;
			}

			uint32_t* const upSizes = m_tmpSizes+1u;
			uint32_t* const downAddresses = m_tmpAddresses+upAllocations;
			uint32_t* const downSizes = m_tmpSizes+upAllocations;

			uint32_t maxDWORDs = 0u;
			uint32_t slabCount = 0u;
			auto SlabComparator = [](auto lhs, auto rhs) -> bool
			{
				if (lhs.contiguousPool==rhs.contiguousPool)
					return lhs.source.begin()<rhs.source.begin();
				return lhs.contiguousPool<rhs.contiguousPool;
			};
			// figure out the sizes to allocate
			{
				m_tmpSizes[0u] = sizeof(nbl_glsl_property_pool_transfer_t)*m_maxPropertiesPerPass;

				uint32_t* upSizesIt = upSizes;
				uint32_t* downSizesIt = downSizes;
				for (uint32_t i=0; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];
					const auto propSize = request.pool->getPropertySize(request.propertyID);
					const auto elementsByteSize = request.indices.size()*propSize;
					maxDWORDs = core::max<uint32_t>(elementsByteSize/sizeof(uint32_t),maxDWORDs);

					if (request.isDownload())
						*(downSizesIt++) = elementsByteSize;
					else
						*(upSizesIt++) = elementsByteSize;

					// if pool is not contiguous, it wont have index redirects, so we dont care about it
					m_tmpIndexRanges[i].contiguousPool = request.pool->isContiguous() ? request.pool:nullptr;
					m_tmpIndexRanges[i].source = request.indices;
					m_tmpIndexRanges[i].destOff = 0u;
				}

				// find slabs = contiguous or repeated ranges of indices (reduce index duplication)
				{
					std::sort(m_tmpIndexRanges,m_tmpIndexRanges+propertiesThisPass,SlabComparator);

					uint32_t indexOffset = 0u;
					auto oit = m_tmpIndexRanges;
					for (auto i=1u; i<propertiesThisPass; i++)
					{
						const auto& inRange = m_tmpIndexRanges[i];

						// check for discontinuity
						auto& outRange = *oit;
						// if two requests have different contiguous pools, they'll have different redirects, so cant merge duplicate index data ranges
						if (inRange.contiguousPool!=outRange.contiguousPool || inRange.source.begin()>outRange.source.end())
						{
							indexOffset += outRange.source.size();
							// begin a new slab
							oit++;
							*oit = inRange;
							oit->destOff = indexOffset;
						}
						else
							outRange.source = {outRange.source.begin(),inRange.source.end()};
					}
					// note the size of the last slab
					indexOffset += oit->source.size();
					slabCount = std::distance(m_tmpIndexRanges,++oit);

					m_tmpSizes[0u] += indexOffset*sizeof(uint32_t);
				}
			}
			// allocate indices and upload/allocate data
			bool retval = true; // success
			std::fill(m_tmpAddresses,m_tmpAddresses+propertiesThisPass+1u,invalid_address);
			std::fill(downAddresses,downAddresses+propertiesThisPass,invalid_address);
			auto freeUpAllocOnFail = core::makeRAIIExiter([&]()->void
			{ 
				if (!retval)
				{
					upBuff->multi_free(upAllocations,m_tmpAddresses,m_tmpSizes);
					upBuff->multi_free(downAllocations,downAddresses,downSizes);
				}
			});
			{
				// TODO: handle overflow (chunk the updates with `max_size()` on the upload and download allocators)
				const auto unallocatedBytes = upBuff->multi_alloc(maxWaitPoint,upAllocations,m_tmpAddresses,m_tmpSizes,m_alignments);
				if (!(retval=unallocatedBytes==0u))
				{
					logger.log("CPropertyPoolHandler: Timed out during upstream staging allocation, failed to allocate %d bytes!",system::ILogger::ELL_ERROR,unallocatedBytes);
					return retval;
				}
			}
			// allocate and write
			{
				IDriverMemoryAllocation::MappedMemoryRange flushRanges[MaxPropertyTransfers+1u];
				flushRanges[0].memory = upBuff->getBuffer()->getBoundMemory();
				flushRanges[0].range = {m_tmpAddresses[0],m_tmpSizes[0]};

				uint32_t* indexBufferPtr = reinterpret_cast<uint32_t*>(upBuffPtr+m_tmpAddresses[0u]);
				// write header
				for (uint32_t i=0; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];

					auto& transfer = reinterpret_cast<nbl_glsl_property_pool_transfer_t*>(indexBufferPtr)[i];
					transfer.propertyDWORDsize_upDownFlag = request.pool->getPropertySize(request.propertyID)/sizeof(uint32_t);
					if (request.isDownload())
						transfer.propertyDWORDsize_upDownFlag = -transfer.propertyDWORDsize_upDownFlag;
					const auto& originalRange = request.indices;
					transfer.elementCount = originalRange.size();
					{
						// find the slab
						IndexUploadRange dummy;
						dummy.contiguousPool = request.pool->isContiguous() ? request.pool:nullptr;
						dummy.source = originalRange;
						dummy.destOff = 0xdeadbeefu;
						auto aboveOrEqual = std::lower_bound(m_tmpIndexRanges,m_tmpIndexRanges+slabCount,dummy,SlabComparator);
						auto containing = aboveOrEqual->source.begin()!=originalRange.begin() ? (aboveOrEqual-1):aboveOrEqual;
						//
						assert((containing->contiguousPool==request.pool||!request.pool->isContiguous()) && containing->source.begin()<=originalRange.begin() && originalRange.end()<=containing->source.end());
						transfer.indexOffset = containing->destOff+(originalRange.begin()-containing->source.begin());
					}
				}
				indexBufferPtr += (sizeof(nbl_glsl_property_pool_transfer_t)/sizeof(uint32_t))*m_maxPropertiesPerPass;
				// write the indices
				for (auto i=0u; i<slabCount; i++)
				{
					const auto& range = m_tmpIndexRanges[i];
					const auto* indices = range.source.begin();
					const auto indexCount = range.source.size();
					if (range.contiguousPool)
					{
						for (auto j=0u; j<indexCount; j++)
							indexBufferPtr[j] = range.contiguousPool->indexToAddress(indices[j]);
					}
					else
						std::copy_n(indices,indexCount,indexBufferPtr);
					indexBufferPtr += indexCount;
				}
	
				// upload
				auto flushRangesIt = flushRanges+1u;
				auto upAddrIt = m_tmpAddresses+1u;
				for (uint32_t i=0u; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];
					if (request.isDownload())
						continue;

					const auto addr = *(upAddrIt++);
				
					*flushRangesIt = flushRanges[0];
					(flushRangesIt++)->range = {addr,upSizes[i]};

					assert(addr!=invalid_address);
					const size_t propSize = request.pool->getPropertySize(request.propertyID);
					memcpy(upBuffPtr+addr,request.data,request.indices.size()*propSize);
				}

				// flush if needed
				if (upBuff->needsManualFlushOrInvalidate())
					m_device->flushMappedMemoryRanges(upAllocations,flushRanges);


				if (downAllocations)
				{
					const auto unallocatedBytes = downBuff->multi_alloc(maxWaitPoint,downAllocations,downAddresses,downSizes,m_alignments);
					if (!(retval=unallocatedBytes==0u))
					{
						logger.log("CPropertyPoolHandler: Timed out during downstream staging allocation, failed to allocate %d bytes!",system::ILogger::ELL_ERROR,unallocatedBytes);
						return retval;
					}
				}
			}

			// update desc sets
			auto setIx = m_dsCache->acquireSet(this,upBuff->getBuffer(),downBuff->getBuffer(),localRequests,propertiesThisPass,m_tmpSizes[0],m_tmpAddresses,downAddresses);
			if (!(retval=setIx!=IDescriptorSetCache::invalid_index))
			{
				logger.log("CPropertyPoolHandler: Failed to acquire descriptor set!",system::ILogger::ELL_ERROR);
				return retval;
			}

			cmdbuf->bindComputePipeline(m_pipeline.get());
			// bind desc sets
			auto set = m_dsCache->getSet(setIx);
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_pipeline->getLayout(),0u,1u,&set,nullptr);
			// dispatch
			const auto xWorkgroups = (maxDWORDs-1u)/IdealWorkGroupSize+1u;
			//if (xWorkgroups>m_device->getPhysicalDevice()->getLimits().maxDispatchSize[0])
				// TODO: log error
			cmdbuf->dispatch(xWorkgroups,propertiesThisPass,1u);

			// deferred release resources
			m_dsCache->releaseSet(m_device.get(),core::smart_refctd_ptr<IGPUFence>(fence),setIx);
			// dont drop the cmdbuffer until the transfer is complete
			auto cmdbuffs = reinterpret_cast<const IGPUCommandBuffer**>(m_tmpIndexRanges);
			cmdbuffs[0] = cmdbuf;
			std::fill_n(cmdbuffs+1u,upAllocations-1u,nullptr);
			upBuff->multi_free(upAllocations,m_tmpAddresses,m_tmpSizes,core::smart_refctd_ptr<IGPUFence>(fence),cmdbuffs);
			result.download.push(downAllocations,downAddresses,downSizes);

			return retval;
		};

		
		auto requests = requestsBegin;
		for (uint32_t i=0; i<fullPasses; i++)
		{
			result.transferSuccess = copyPass(requests,m_maxPropertiesPerPass)&&result.transferSuccess;
			requests += m_maxPropertiesPerPass;
		}

		const auto leftOverProps = totalProps-fullPasses*m_maxPropertiesPerPass;
		if (leftOverProps)
			result.transferSuccess = copyPass(requests,leftOverProps)&&result.transferSuccess;
	}
	return result;
}

CPropertyPoolHandler::download_future_t::~download_future_t()
{
	if (m_allocCount && m_sizes)
	{
		const auto status = m_device->getFenceStatus(m_fence.get());
		if (status==IGPUFence::ES_TIMEOUT||status==IGPUFence::ES_NOT_READY)
		{
			const core::IReferenceCounted*const *const perfectForwardingAintPerfect = nullptr;
			m_downBuff->multi_free(m_allocCount,m_addresses,m_sizes,std::move(m_fence),perfectForwardingAintPerfect);
		}
		else
			m_downBuff->multi_free(m_allocCount,m_addresses,m_sizes);
	}
	if (m_addresses)
		delete[] m_addresses;
}

bool CPropertyPoolHandler::download_future_t::wait()
{
	if (!m_allocCount || !m_sizes)
		return true;

	IGPUFence::E_STATUS result;
	do
	{
		result = m_device->waitForFences(1u,&m_fence.get(),true,9999999999ull);
	} while (result==IGPUFence::ES_TIMEOUT||result==IGPUFence::ES_NOT_READY);
	m_downBuff->multi_free(m_allocCount,m_addresses,m_sizes);
	m_sizes = nullptr;
	m_fence = nullptr;
	m_device = nullptr;
	return result==IGPUFence::ES_SUCCESS;
}

const void* CPropertyPoolHandler::download_future_t::getData(const uint32_t downRequestIndex)
{
	if (downRequestIndex<m_allocCount)
		return reinterpret_cast<uint8_t*>(m_downBuff->getBufferPointer())+m_addresses[downRequestIndex];
	return nullptr;
}


uint32_t CPropertyPoolHandler::TransferDescriptorSetCache::acquireSet(
	CPropertyPoolHandler* handler,
	IGPUBuffer* const upBuff,
	IGPUBuffer* const downBuff,
	const TransferRequest* requests,
	const uint32_t propertyCount,
	const uint32_t firstSSBOSize,
	const uint32_t* uploadAddresses,
	const uint32_t* downloadAddresses
)
{
	auto retval = IDescriptorSetCache::acquireSet();
	if (retval==IDescriptorSetCache::invalid_index)
		return IDescriptorSetCache::invalid_index;
	IGPUDescriptorSet* set = IDescriptorSetCache::getSet(retval);
	
	//
	auto device = handler->getDevice();

	const auto maxPropertiesPerPass = handler->getMaxPropertiesPerTransferDispatch();
	IGPUDescriptorSet::SDescriptorInfo infos[MaxPropertyTransfers*2u+1u];
	infos[0].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
	infos[0].buffer = { *(uploadAddresses++),firstSSBOSize };
	auto* inDescInfo = infos+1;
	auto* outDescInfo = infos+1+maxPropertiesPerPass;
	for (uint32_t i=0u; i<propertyCount; i++)
	{
		const auto& request = requests[i];
			
		const auto* pool = request.pool;
		const auto& propMemBlock = pool->getPropertyMemoryBlock(request.propertyID);
		const uint32_t transferPropertySize = request.indices.size()*pool->getPropertySize(request.propertyID);

		if (request.isDownload())
		{
			inDescInfo[i].desc = core::smart_refctd_ptr<asset::IDescriptor>(propMemBlock.buffer);
			inDescInfo[i].buffer = {propMemBlock.offset,propMemBlock.size};

			outDescInfo[i].desc = core::smart_refctd_ptr<asset::IDescriptor>(downBuff);
			outDescInfo[i].buffer = { *(downloadAddresses++),transferPropertySize };
		}
		else
		{
			inDescInfo[i].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
			inDescInfo[i].buffer = { *(uploadAddresses++),transferPropertySize };
					
			outDescInfo[i].desc = core::smart_refctd_ptr<asset::IDescriptor>(propMemBlock.buffer);
			outDescInfo[i].buffer = {propMemBlock.offset,propMemBlock.size};
		}
	}
	{
		auto assignBuf = [](auto& info, auto streamBuff) -> void
		{
			info.desc = core::smart_refctd_ptr<IGPUBuffer>(streamBuff);
			info.buffer = {0u,streamBuff->getSize()};
		};
		for (auto i=propertyCount; i<maxPropertiesPerPass; i++)
			assignBuf(inDescInfo[i],upBuff);
		for (auto i=propertyCount; i<maxPropertiesPerPass; i++)
			assignBuf(outDescInfo[i],downBuff);
	}
	IGPUDescriptorSet::SWriteDescriptorSet writes[3u];
	for (auto i=0u; i<3u; i++)
	{
		writes[i].dstSet = set;
		writes[i].binding = i;
		writes[i].arrayElement = 0u;
		writes[i].count = i ? maxPropertiesPerPass:1u;
		writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
		writes[i].info = i ? (writes[i-1u].info+writes[i-1u].count):infos;
	}
	device->updateDescriptorSets(3u,writes,0u,nullptr);

	return retval;
}