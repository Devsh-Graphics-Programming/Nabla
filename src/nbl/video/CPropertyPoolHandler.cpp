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
	
	const auto maxStreamingAllocations = m_maxPropertiesPerPass+1u;
	{
		m_tmpIndexRanges = reinterpret_cast<IndexUploadRange*>(malloc((sizeof(IndexUploadRange)+sizeof(nbl_glsl_property_pool_transfer_t))*maxStreamingAllocations));
		m_tmpAddresses = reinterpret_cast<uint32_t*>(m_tmpIndexRanges+maxStreamingAllocations);
		m_tmpSizes = reinterpret_cast<uint32_t*>(m_tmpAddresses+maxStreamingAllocations);
		m_alignments = reinterpret_cast<uint32_t*>(m_tmpSizes+maxStreamingAllocations);
		std::fill_n(m_alignments,maxStreamingAllocations,alignof(uint32_t));
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
bool CPropertyPoolHandler::addProperties(IGPUCommandBuffer* cmdbuf, const AllocationRequest* requestsBegin, const AllocationRequest* requestsEnd)
{
	bool success = true;

	uint32_t transferCount = 0u;
	for (auto it=requestsBegin; it!=requestsEnd; it++)
	{
		success = it->pool->allocateProperties(it->outIndices.begin(),it->outIndices.end()) && success;
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
	return transferProperties(cmdbuf,transferRequests.data(),transferRequests.data()+transferCount) && success;
}

//
bool CPropertyPoolHandler::transferProperties(IGPUCommandBuffer* cmdbuf, const TransferRequest* requestsBegin, const TransferRequest* requestsEnd)
{
	const auto totalProps = std::distance(requestsBegin,requestsEnd);

	bool retval = true;
	if (totalProps!=0u)
	{
		const auto fullPasses = totalProps/m_maxPropertiesPerPass;

		auto upBuff = m_device->getDefaultUpStreamingBuffer();
		auto downBuff = m_device->getDefaultDownStreamingBuffer();
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
					m_tmpIndexRanges.resize(std::distance(m_tmpIndexRanges,++oit));

					m_tmpSizes[0u] += indexOffset*sizeof(uint32_t);
				}
			}
			// allocate indices and upload/allocate data
			{
				// TODO: handle overflow (chunk the updates)
				std::fill(m_tmpAddresses,m_tmpAddresses+propertiesThisPass+1u,invalid_address);
				upBuff->multi_alloc(maxWaitPoint,upAllocations,m_tmpAddresses,m_tmpSizes,m_alignments);
#if 0

				uint8_t* indexBufferPtr = upBuffPtr+m_tmpAddresses[0u]/sizeof(uint32_t);
				// write `elementCount`
				for (uint32_t i=0; i<propertiesThisPass; i++)
					*(indexBufferPtr++) = localRequests[i].indices.size();
				// write `propertyDWORDsize_upDownFlag`
				for (uint32_t i=0; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];
					int32_t propSize = request.pool->getPropertySize(request.propertyID);
					propSize /= sizeof(uint32_t);
					if (request.download)
						propSize = -propSize;
					*reinterpret_cast<int32_t*>(indexBufferPtr++) = propSize;
				}

				// write `indexOffset`
				for (uint32_t i=0; i<propertiesThisPass; i++)
				{
					const auto& originalRange = localRequests->indices;
					// find the slab
					IndexUploadRange dummy;
					dummy.source = originalRange;
					dummy.destOff = 0xdeadbeefu;
					auto aboveOrEqual = std::lower_bound(m_tmpIndexRanges.begin(),m_tmpIndexRanges.end(),dummy,RangeComparator);
					auto containing = aboveOrEqual->source.begin()!=originalRange.begin() ? (aboveOrEqual-1):aboveOrEqual;
					//
					assert(containing->source.begin()<=originalRange.begin() && originalRange.end()<=containing->source.end());
					*(indexBufferPtr++) = containing->destOff+(originalRange.begin()-containing->source.begin());
				}
				// write the indices
				for (auto slab : m_tmpIndexRanges)
				{
					const auto indexCount = slab.source.size();
					memcpy(indexBufferPtr,slab.source.begin(),sizeof(uint32_t)*indexCount);
					indexBufferPtr += indexCount;
				}
				
				// upload
				auto upAddrIt = m_tmpAddresses.begin()+1;
				for (uint32_t i=0u; i<propertiesThisPass; i++)
				{
					const auto& request = localRequests[i];
					if (request.download)
						continue;
					
					if ((*upAddrIt)!=invalid_address)
					{
						size_t propSize = request.pool->getPropertySize(request.propertyID);
						memcpy(upBuffPtr+(*(upAddrIt++)),request.writeData,request.indices.size()*propSize);
					}
				}
#endif

				if (downAllocations)
					downBuff->multi_alloc(maxWaitPoint,downAllocations,downAddresses,downSizes,m_alignments);
			}

			cmdbuf->bindComputePipeline(m_pipeline.get());
			// update desc sets
			auto set = m_dsCache->acquireSet(this,localRequests,propertiesThisPass,m_tmpSizes[0],m_tmpAddresses,downAddresses);
			if (!set)
				return false;

			// bind desc sets
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,m_pipeline->getLayout(),0u,1u,&set,nullptr);
			// dispatch
			cmdbuf->dispatch((maxElements-1u)/IdealWorkGroupSize+1u,propertiesThisPass,1u);

#if 0
			auto& fence = retval.second = m_driver->placeFence(true);
			// deferred release resources
			items.descriptorSetCache.releaseSet(core::smart_refctd_ptr(fence),std::move(set));
			upBuff->multi_free(upAllocations,m_tmpAddresses.data(),m_tmpSizes.data(),core::smart_refctd_ptr(fence));
			if (downAllocations)
				downBuff->multi_free(downAllocations,downAddresses,downSizes,core::smart_refctd_ptr(fence));
#endif
			return true;
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

IGPUDescriptorSet* CPropertyPoolHandler::TransferDescriptorSetCache::acquireSet(
	CPropertyPoolHandler* handler,
	const TransferRequest* requests,
	const uint32_t indexCount,
	const uint32_t propertyCount,
	const uint32_t* uploadAddresses,
	const uint32_t* downloadAddresses
)
{
	auto setIx = IDescriptorSetCache::acquireSet();
	if (setIx==IDescriptorSetCache::invalid_index)
		return nullptr;
	IGPUDescriptorSet* retval = IDescriptorSetCache::getSet(setIx);
	
	//
	auto device = handler->getDevice();
	auto upBuff = device->getDefaultUpStreamingBuffer()->getBuffer();
	auto downBuff = device->getDefaultDownStreamingBuffer()->getBuffer();

	const auto maxPropertiesPerPass = handler->getMaxPropertiesPerTransferDispatch();
	IGPUDescriptorSet::SDescriptorInfo infos[MaxPropertyTransfers*2u+1u];
	infos[0].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
	infos[0].buffer = { *(uploadAddresses++),sizeof(nbl_glsl_property_pool_transfer_t)*maxPropertiesPerPass+indexCount*sizeof(uint32_t) };
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
		writes[i].dstSet = retval;
		writes[i].binding = i;
		writes[i].arrayElement = 0u;
		writes[i].count = i ? maxPropertiesPerPass:1u;
		writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
		writes[i].info = i ? (writes[i-1u].info+writes[i-1u].count):infos;
	}
	device->updateDescriptorSets(3u,writes,0u,nullptr);

	return retval;
}