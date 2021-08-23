#include "nbl/video/IPropertyPool.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

#include "nbl/builtin/glsl/property_pool/transfer.glsl"

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
		std::fill_n(m_alignments,maxStreamingAllocations,static_cast<uint32_t>(alignof(uint32_t)));
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

//
bool CPropertyPoolHandler::addProperties(
	IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence,
	const AllocationRequest* const requestsBegin, const AllocationRequest* const requestsEnd,
	system::logger_opt_ptr logger, const std::chrono::high_resolution_clock::time_point maxWaitPoint
)
{
	return addProperties(m_device->getDefaultUpStreamingBuffer(),m_device->getDefaultDownStreamingBuffer(),cmdbuf,fence,requestsBegin,requestsEnd,logger,maxWaitPoint);
}
bool CPropertyPoolHandler::addProperties(
	StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
	IGPUFence* const fence, const AllocationRequest* const requestsBegin, const AllocationRequest* const requestsEnd, system::logger_opt_ptr logger,
	const std::chrono::high_resolution_clock::time_point maxWaitPoint=std::chrono::high_resolution_clock::now()+std::chrono::microseconds(1500u)
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
		oit->download = false;
		oit->pool = it->pool;
		oit->indices = { it->outIndices.begin(),it->outIndices.end() };
		oit->propertyID = i;
		oit->readData = it->data[i];
		oit++;
	}
	return transferProperties(upBuff,downBuff,cmdbuf,fence,transferRequests.data(),transferRequests.data()+transferCount,logger,maxWaitPoint) && success;
}

//
bool CPropertyPoolHandler::transferProperties(
	IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence,
	const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd,
	system::logger_opt_ptr logger, const std::chrono::high_resolution_clock::time_point maxWaitPoint
)
{
	return transferProperties(m_device->getDefaultUpStreamingBuffer(),m_device->getDefaultDownStreamingBuffer(),cmdbuf,fence,requestsBegin,requestsEnd,logger,maxWaitPoint);
}
bool CPropertyPoolHandler::transferProperties(
	StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
	IGPUFence* const fence, const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd, system::logger_opt_ptr logger,
	const std::chrono::high_resolution_clock::time_point maxWaitPoint
)
{
	const auto totalProps = std::distance(requestsBegin,requestsEnd);

	bool retval = true;
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
				if (localRequests[i].download)
					downAllocations++;
				else
					upAllocations++;
			}

			uint32_t* const upSizes = m_tmpSizes+1u;
			uint32_t* const downAddresses = m_tmpAddresses+upAllocations;
			uint32_t* const downSizes = m_tmpSizes+upAllocations;

			uint32_t maxElements = 0u;
			uint32_t slabCount = 0u;
			auto RangeComparator = [](auto lhs, auto rhs)->bool{return lhs.source.begin()<rhs.source.begin();};
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

					if (request.download)
						*(downSizesIt++) = elementsByteSize;
					else
						*(upSizesIt++) = elementsByteSize;

					m_tmpIndexRanges[i].source = request.indices;
					m_tmpIndexRanges[i].destOff = 0u;
				}

				// find slabs (reduce index duplication)
				{
					std::sort(m_tmpIndexRanges,m_tmpIndexRanges+propertiesThisPass,RangeComparator);

					uint32_t indexOffset = 0u;
					auto oit = m_tmpIndexRanges;
					for (auto i=1u; i<propertiesThisPass; i++)
					{
						const auto& inRange = m_tmpIndexRanges[i].source;
						maxElements = core::max<uint32_t>(inRange.size(),maxElements);

						// check for discontinuity
						auto& outRange = oit->source;
						if (inRange.begin()>outRange.end())
						{
							indexOffset += outRange.size();
							// begin a new slab
							oit++;
							*oit = m_tmpIndexRanges[i];
							oit->destOff = indexOffset;
						}
						else
							reinterpret_cast<const uint32_t**>(&outRange)[1] = inRange.end();
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
			// upload data
			{
				uint8_t* indexBufferPtr = upBuffPtr+m_tmpAddresses[0u];
				// write header
				for (uint32_t i=0; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];

					auto& transfer = reinterpret_cast<nbl_glsl_property_pool_transfer_t*>(indexBufferPtr)[i];
					transfer.propertyDWORDsize_upDownFlag = request.pool->getPropertySize(request.propertyID)/sizeof(uint32_t);
					if (request.download)
						transfer.propertyDWORDsize_upDownFlag = -transfer.propertyDWORDsize_upDownFlag;
					const auto& originalRange = request.indices;
					transfer.elementCount = originalRange.size();
					{
						// find the slab
						IndexUploadRange dummy;
						dummy.source = originalRange;
						dummy.destOff = 0xdeadbeefu;
						auto aboveOrEqual = std::lower_bound(m_tmpIndexRanges,m_tmpIndexRanges+slabCount,dummy,RangeComparator);
						auto containing = aboveOrEqual->source.begin()!=originalRange.begin() ? (aboveOrEqual-1):aboveOrEqual;
						//
						assert(containing->source.begin()<=originalRange.begin() && originalRange.end()<=containing->source.end());
						transfer.indexOffset = containing->destOff+(originalRange.begin()-containing->source.begin());
					}

					indexBufferPtr += sizeof(nbl_glsl_property_pool_transfer_t);
				}
				// write the indices
				for (auto i=0u; i<slabCount; i++)
				{
					const auto& indices = m_tmpIndexRanges[i].source;
					const auto indexCount = indices.size();
					memcpy(indexBufferPtr,indices.begin(),sizeof(uint32_t)*indexCount);
					indexBufferPtr += indexCount;
				}
	
				// upload
				auto upAddrIt = m_tmpAddresses+1u;
				for (uint32_t i=0u; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];
					if (request.download)
						continue;
					
					const auto addr = *(upAddrIt++);
					assert(addr!=invalid_address);
					const size_t propSize = request.pool->getPropertySize(request.propertyID);
					memcpy(upBuffPtr+addr,request.writeData,request.indices.size()*propSize);
				}

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
			cmdbuf->dispatch((maxElements-1u)/IdealWorkGroupSize+1u,propertiesThisPass,1u);

			// deferred release resources
			m_dsCache->releaseSet(m_device.get(),core::smart_refctd_ptr<IGPUFence>(fence),setIx);
			// dont drop the cmdbuffer until the transfer is complete
			upBuff->multi_free(upAllocations,m_tmpAddresses,m_tmpSizes,core::smart_refctd_ptr<IGPUFence>(fence),&cmdbuf);
			if (downAllocations)
				downBuff->multi_free(downAllocations,downAddresses,downSizes,core::smart_refctd_ptr<IGPUFence>(fence),&cmdbuf);

			return retval;
		};

		
		auto requests = requestsBegin;
		for (uint32_t i=0; i<fullPasses; i++)
		{
			retval = copyPass(requests,m_maxPropertiesPerPass)&&retval;
			requests += m_maxPropertiesPerPass;
		}

		const auto leftOverProps = totalProps-fullPasses*m_maxPropertiesPerPass;
		if (leftOverProps)
			retval = copyPass(requests,leftOverProps)&&retval;
	}
	return retval;
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
		const bool download = request.download;
			
		const auto* pool = request.pool;
		const auto& propMemBlock = pool->getPropertyMemoryBlock(request.propertyID);
		const uint32_t transferPropertySize = request.indices.size()*pool->getPropertySize(request.propertyID);

		if (download)
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