#include "nbl/video/ILogicalDevice.h"

using namespace nbl;
using namespace video;

#if 0
//
constexpr char* copyCsSource = R"(
layout(local_size_x=_NBL_BUILTIN_PROPERTY_COPY_GROUP_SIZE_) in;

layout(set=0,binding=0) readonly restrict buffer Indices
{
    uint elementCount[_NBL_BUILTIN_PROPERTY_COUNT_];
	int propertyDWORDsize_upDownFlag[_NBL_BUILTIN_PROPERTY_COUNT_];
    uint indexOffset[_NBL_BUILTIN_PROPERTY_COUNT_];
    uint indices[];
};


layout(set=0, binding=1) readonly restrict buffer InData
{
    uint data[];
} inBuff[_NBL_BUILTIN_PROPERTY_COUNT_];
layout(set=0, binding=2) writeonly restrict buffer OutData
{
    uint data[];
} outBuff[_NBL_BUILTIN_PROPERTY_COUNT_];


#if 0 // optimization
uint shared workgroupShared[_NBL_BUILTIN_PROPERTY_COPY_GROUP_SIZE_];
#endif


void main()
{
    const uint propID = gl_WorkGroupID.y;

	const int combinedFlag = propertyDWORDsize_upDownFlag[propID];
	const bool download = combinedFlag<0;

	const uint propDWORDs = uint(download ? (-combinedFlag):combinedFlag);
#if 0 // optimization
	const uint localIx = gl_LocalInvocationID.x;
	const uint MaxItemsToProcess = ;
	if (localIx<MaxItemsToProcess)
		workgroupShared[localIx] = indices[localIx+indexOffset[propID]];
	barrier();
	memoryBarrier();
#endif

    const uint index = gl_GlobalInvocationID.x/propDWORDs;
    if (index>=elementCount[propID])
        return;

	const uint redir = (
#if 0 //optimization
		workgroupShared[index]
#else 
		indices[index+indexOffset[propID]]
#endif
	// its equivalent to `indices[index]*propDWORDs+gl_GlobalInvocationID.x%propDWORDs`
    -index)*propDWORDs+gl_GlobalInvocationID.x;

    const uint inIndex = download ? redir:gl_GlobalInvocationID.x;
    const uint outIndex = download ? gl_GlobalInvocationID.x:redir;
	outBuff[propID].data[outIndex] = inBuff[propID].data[inIndex];
}
)";
#endif
//
CPropertyPoolHandler::CPropertyPoolHandler(core::smart_refctd_ptr<ILogicalDevice>&& device) : m_device(std::move(device))
{
	auto system = m_device->getPhysicalDevice()->getSystem();
	
	const auto maxSSBO = m_device->getPhysicalDevice()->getLimits().maxPerStageSSBOs;

	const uint32_t maxPropertiesPerPass = (maxSSBO-1u)/2u;
#if 0
	m_perPropertyCountItems.reserve(maxPropertiesPerPass);
	m_tmpIndexRanges.reserve(maxPropertiesPerPass);

	const auto maxStreamingAllocations = maxPropertiesPerPass+1u;
	m_tmpAddresses.resize(maxSteamingAllocations);
	m_tmpSizes.resize(maxSteamingAllocations);
	m_alignments.resize(maxSteamingAllocations,alignof(uint32_t));

	for (uint32_t i=0u; i<maxPropertiesPerPass; i++)
	{
		const auto propCount = i+1u;
		m_perPropertyCountItems.emplace_back(m_driver,pipelineCache,propCount);
	}
#endif
}

//
CPropertyPoolHandler::transfer_result_t CPropertyPoolHandler::addProperties(const AllocationRequest* requestsBegin, const AllocationRequest* requestsEnd, const std::chrono::high_resolution_clock::time_point& maxWaitPoint)
{
#if 0
	bool success = true;

	uint32_t transferCount = 0u;
	for (auto it=requestsBegin; it!=requestsEnd; it++)
	{
		success = it->pool->allocateProperties(it->outIndices.begin(),it->outIndices.end()) && success;
		transferCount += it->pool->getPropertyCount();
	}

	if (!success)
		return {false,nullptr};

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
	return transferProperties(transferRequests.data(),transferRequests.data()+transferCount,maxWaitPoint);
#endif
	return {false,nullptr};
}

//
CPropertyPoolHandler::transfer_result_t CPropertyPoolHandler::transferProperties(const TransferRequest* requestsBegin, const TransferRequest* requestsEnd, const std::chrono::high_resolution_clock::time_point& maxWaitPoint)
{
	const auto totalProps = std::distance(requestsBegin,requestsEnd);

	transfer_result_t retval = { true,nullptr };
#if 0
	if (totalProps!=0u)
	{
		const uint32_t maxPropertiesPerPass = m_perPropertyCountItems.size();
		const auto fullPasses = totalProps/maxPropertiesPerPass;

		auto upBuff = m_driver->getDefaultUpStreamingBuffer();
		auto downBuff = m_driver->getDefaultDownStreamingBuffer();
		constexpr auto invalid_address = std::remove_reference_t<decltype(upBuff->getAllocator())>::invalid_address;
		uint8_t* upBuffPtr = reinterpret_cast<uint8_t*>(upBuff->getBufferPointer());
				
		auto copyPass = [&](const TransferRequest* localRequests, uint32_t propertiesThisPass) -> void
		{
			const uint32_t headerSize = sizeof(uint32_t)*3u*propertiesThisPass;

			uint32_t upAllocations = 1u;
			uint32_t downAllocations = 0u;
			for (uint32_t i=0u; i<propertiesThisPass; i++)
			{
				if (localRequests[i].download)
					downAllocations++;
				else
					upAllocations++;
			}
			
			uint32_t* const upSizes = m_tmpSizes.data()+1u;
			uint32_t* const downAddresses = m_tmpAddresses.data()+upAllocations;
			uint32_t* const downSizes = m_tmpSizes.data()+upAllocations;

			// figure out the sizes to allocate
			uint32_t maxElements = 0u;
			auto RangeComparator = [](auto lhs, auto rhs)->bool{return lhs.source.begin()<rhs.source.begin();};
			{
				m_tmpSizes[0u] = 3u*propertiesThisPass;

				uint32_t* upSizesIt = upSizes;
				uint32_t* downSizesIt = downSizes;
				m_tmpIndexRanges.resize(propertiesThisPass);
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
				std::sort(m_tmpIndexRanges.begin(),m_tmpIndexRanges.end(),RangeComparator);
				uint32_t indexOffset = 0u;
				auto oit = m_tmpIndexRanges.begin();
				for (auto it=m_tmpIndexRanges.begin()+1u; it!=m_tmpIndexRanges.end(); it++)
				{
					const auto& inRange = it->source;
					maxElements = core::max<uint32_t>(inRange.size(),maxElements);

					// check for discontinuity
					auto& outRange = oit->source;
					if (inRange.begin()>outRange.end())
					{
						indexOffset += outRange.size();
						// begin a new slab
						oit++;
						*oit = *it;
						oit->destOff = indexOffset;
					}
					else
						reinterpret_cast<const uint32_t**>(&outRange)[1] = inRange.end();
				}
				// note the size of the last slab
				indexOffset += oit->source.size();
				m_tmpIndexRanges.resize(std::distance(m_tmpIndexRanges.begin(),++oit));

				m_tmpSizes[0u] += indexOffset;
				m_tmpSizes[0u] *= sizeof(uint32_t);
			}

			// allocate indices and upload/allocate data
			{
				std::fill(m_tmpAddresses.begin(),m_tmpAddresses.begin()+propertiesThisPass+1u,invalid_address);
				upBuff->multi_alloc(maxWaitPoint,upAllocations,m_tmpAddresses.data(),m_tmpSizes.data(),m_alignments.data());

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

				if (downAllocations)
					downBuff->multi_alloc(maxWaitPoint,downAllocations,downAddresses,downSizes,m_alignments.data());
			}

			const auto pipelineIndex = propertiesThisPass-1u;
			auto& items = m_perPropertyCountItems[pipelineIndex];
			auto pipeline = items.pipeline.get();
			m_driver->bindComputePipeline(pipeline);

			// update desc sets
			auto set = items.descriptorSetCache.getNextSet(m_driver,localRequests,m_tmpSizes[0],m_tmpAddresses.data(),downAddresses);
			if (!set)
			{
				retval.first = false;
				return;
			}

			// bind desc sets
			m_driver->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&set.get(),nullptr);
		
			// dispatch (this will need to change to a cmd buffer submission with a fence)
			m_driver->dispatch((maxElements+IdealWorkGroupSize-1u)/IdealWorkGroupSize,propertiesThisPass,1u);
			auto& fence = retval.second = m_driver->placeFence(true);

			// deferred release resources
			upBuff->multi_free(upAllocations,m_tmpAddresses.data(),m_tmpSizes.data(),core::smart_refctd_ptr(fence));
			if (downAllocations)
				downBuff->multi_free(downAllocations,downAddresses,downSizes,core::smart_refctd_ptr(fence));
			items.descriptorSetCache.releaseSet(core::smart_refctd_ptr(fence),std::move(set));
		};

		
		auto requests = requestsBegin;
		for (uint32_t i=0; i<fullPasses; i++)
		{
			copyPass(requests,maxPropertiesPerPass);
			requests += maxPropertiesPerPass;
		}

		const auto leftOverProps = totalProps-fullPasses*maxPropertiesPerPass;
		if (leftOverProps)
			copyPass(requests,leftOverProps);
	}
#endif
	return retval;
}

#if 0
//
CPropertyPoolHandler::PerPropertyCountItems::PerPropertyCountItems(IVideoDriver* driver, IGPUPipelineCache* pipelineCache, uint32_t propertyCount) : descriptorSetCache(driver,propertyCount)
{
	std::string shaderSource("#version 440 core\n");
	// property count
	shaderSource += "#define _NBL_BUILTIN_PROPERTY_COUNT_ ";
	shaderSource += std::to_string(propertyCount)+"\n";
	// workgroup sizes
	shaderSource += "#define _NBL_BUILTIN_PROPERTY_COPY_GROUP_SIZE_ ";
	shaderSource += std::to_string(IdealWorkGroupSize)+"\n";
	//
	shaderSource += copyCsSource;

	auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSource.c_str());

	auto shader = driver->createGPUShader(std::move(cpushader));
	auto specshader = driver->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});

	auto layout = driver->createGPUPipelineLayout(nullptr,nullptr,descriptorSetCache.getLayout());
	pipeline = driver->createGPUComputePipeline(pipelineCache,std::move(layout),std::move(specshader));
}


//
CPropertyPoolHandler::DescriptorSetCache::DescriptorSetCache(IVideoDriver* driver, uint32_t _propertyCount) : propertyCount(_propertyCount)
{
	IGPUDescriptorSetLayout::SBinding bindings[3];
	for (auto j=0; j<3; j++)
	{
		bindings[j].binding = j;
		bindings[j].type = asset::EDT_STORAGE_BUFFER;
		bindings[j].count = j ? propertyCount:1u;
		bindings[j].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bindings[j].samplers = nullptr;
	}
	layout = driver->createGPUDescriptorSetLayout(bindings,bindings+3);
	unusedSets.reserve(4u); // 4 frames worth at least
}

CPropertyPoolHandler::DeferredDescriptorSetReclaimer::single_poll_t CPropertyPoolHandler::DeferredDescriptorSetReclaimer::single_poll;
core::smart_refctd_ptr<IGPUDescriptorSet> CPropertyPoolHandler::DescriptorSetCache::getNextSet(
	IVideoDriver* driver, const TransferRequest* requests, uint32_t parameterBufferSize, const uint32_t* uploadAddresses, const uint32_t* downloadAddresses
)
{
	deferredReclaims.pollForReadyEvents(DeferredDescriptorSetReclaimer::single_poll);

	core::smart_refctd_ptr<IGPUDescriptorSet> retval;
	if (unusedSets.size())
	{
		retval = std::move(unusedSets.back());
		unusedSets.pop_back();
	}
	else
		retval = driver->createGPUDescriptorSet(core::smart_refctd_ptr(layout));


	constexpr auto kSyntheticMax = 64;
	assert(propertyCount<kSyntheticMax);
	IGPUDescriptorSet::SDescriptorInfo info[kSyntheticMax];

	IGPUDescriptorSet::SWriteDescriptorSet dsWrite[3u];
	{
		auto upBuff = driver->getDefaultUpStreamingBuffer()->getBuffer();
		auto downBuff = driver->getDefaultDownStreamingBuffer()->getBuffer();

		info[0].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
		info[0].buffer = { *(uploadAddresses++),parameterBufferSize };
		for (uint32_t i=0u; i<propertyCount; i++)
		{
			const auto& request = requests[i];

			const bool download = request.download;
			
			const auto* pool = request.pool;
			const auto& poolMemBlock = pool->getMemoryBlock();

			const uint32_t propertySize = pool->getPropertySize(request.propertyID);
			const uint32_t transferPropertySize = request.indices.size()*propertySize;
			const uint32_t poolPropertyBlockSize = pool->getCapacity()*propertySize;

			auto& inDescInfo = info[i+1];
			auto& outDescInfo = info[propertyCount+i+1];
			if (download)
			{
				inDescInfo.desc = core::smart_refctd_ptr<asset::IDescriptor>(poolMemBlock.buffer);
				inDescInfo.buffer = { pool->getPropertyOffset(request.propertyID),poolPropertyBlockSize };

				outDescInfo.desc = core::smart_refctd_ptr<asset::IDescriptor>(downBuff);
				outDescInfo.buffer = { *(downloadAddresses++),transferPropertySize };
			}
			else
			{
				inDescInfo.desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
				inDescInfo.buffer = { *(uploadAddresses++),transferPropertySize };
					
				outDescInfo.desc = core::smart_refctd_ptr<asset::IDescriptor>(poolMemBlock.buffer);
				outDescInfo.buffer = { pool->getPropertyOffset(request.propertyID),poolPropertyBlockSize };
			}
		}
	}
	for (auto i=0u; i<3u; i++)
	{
		dsWrite[i].dstSet = retval.get();
		dsWrite[i].binding = i;
		dsWrite[i].arrayElement = 0u;
		dsWrite[i].descriptorType = asset::EDT_STORAGE_BUFFER;
	}
	dsWrite[0].count = 1u;
	dsWrite[0].info = info+0;
	dsWrite[1].count = propertyCount;
	dsWrite[1].info = info+1;
	dsWrite[2].count = propertyCount;
	dsWrite[2].info = info+1+propertyCount;
	driver->updateDescriptorSets(3u,dsWrite,0u,nullptr);

	return retval;
}

void CPropertyPoolHandler::DescriptorSetCache::releaseSet(core::smart_refctd_ptr<IDriverFence>&& fence, core::smart_refctd_ptr<IGPUDescriptorSet>&& set)
{
	deferredReclaims.addEvent(GPUEventWrapper(std::move(fence)),DeferredDescriptorSetReclaimer(&unusedSets,std::move(set)));
}
#endif