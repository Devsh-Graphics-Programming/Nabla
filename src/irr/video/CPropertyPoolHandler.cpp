#include "irr/video/CPropertyPoolHandler.h"
#include "irr/video/CPropertyPool.h"

using namespace irr;
using namespace video;

//
constexpr char* copyCsSource = R"(
layout(local_size_x=_IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_) in;

layout(set=0,binding=0) readonly restrict buffer Indices
{
    uint elementCount[_IRR_BUILTIN_PROPERTY_COUNT_];
	int propertyDWORDsize_upDownFlag[_IRR_BUILTIN_PROPERTY_COUNT_];
    uint indexOffset[_IRR_BUILTIN_PROPERTY_COUNT_];
    uint indices[];
};


layout(set=0, binding=1) readonly restrict buffer InData
{
    uint data[];
} inBuff[_IRR_BUILTIN_PROPERTY_COUNT_];
layout(set=0, binding=2) writeonly restrict buffer OutData
{
    uint data[];
} outBuff[_IRR_BUILTIN_PROPERTY_COUNT_];


#if 0 // optimization
uint shared workgroupShared[_IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_];
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

//
CPropertyPoolHandler::CPropertyPoolHandler(IVideoDriver* driver, IGPUPipelineCache* pipelineCache) : m_driver(driver)
{
	assert(m_driver);
	const auto maxSSBO = m_driver->getMaxSSBOBindings(); // TODO: make sure not dynamic offset
	const uint32_t maxPropertiesPerPass = (maxSSBO-1u)/2u;

	m_perPropertyCountItems.reserve(maxPropertiesPerPass);
	m_tmpSizes.resize(maxPropertiesPerPass);
	m_alignments.resize(maxPropertiesPerPass,alignof(uint32_t));

	for (uint32_t i=0u; i<maxPropertiesPerPass; i++)
	{
		const auto propCount = i+1u;
		m_perPropertyCountItems.emplace_back(m_driver,pipelineCache,propCount);
	}
}

//
bool CPropertyPoolHandler::addProperties(IPropertyPool* const* poolsBegin, IPropertyPool* const* poolsEnd, uint32_t* const* indicesBegin, uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd)
{
	bool success = true;

	auto poolIndicesBegin = indicesBegin;
	auto poolIndicesEnd = indicesEnd;
	for (auto it=poolsBegin; it!=poolsEnd; it++)
	{
		success = (*it)->allocateProperties(*poolIndicesBegin,*poolIndicesEnd) && success;
		poolIndicesBegin++;
		poolIndicesEnd++;
	}

	return uploadProperties(poolsBegin,poolsEnd,indicesBegin,indicesEnd,dataBegin,dataEnd) && success;
}


//
CPropertyPoolHandler::PerPropertyCountItems::PerPropertyCountItems(IVideoDriver* driver, IGPUPipelineCache* pipelineCache, uint32_t propertyCount)
{
	std::string shaderSource("#version 440 core\n");
	// property count
	shaderSource += "#define _IRR_BUILTIN_PROPERTY_COUNT_ ";
	shaderSource += std::to_string(propertyCount)+"\n";
	// workgroup sizes
	shaderSource += "#define _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_ ";
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

CPropertyPoolHandler::DescriptorSetCache::DeferredDescriptorSetReclaimer::single_poll_t CPropertyPoolHandler::DescriptorSetCache::DeferredDescriptorSetReclaimer::single_poll;
core::smart_refctd_ptr<IGPUDescriptorSet> CPropertyPoolHandler::DescriptorSetCache::getNextSet(
	IVideoDriver* driver, uint32_t indexByteOffset, uint32_t indexByteSize,
	const uint32_t* uploadByteOffsets, const uint32_t* uploadByteSizes,
	const uint32_t* downloadByteOffets, const uint32_t* downloadByteSize
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

		uint32_t ix = 0u;
		info[ix].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
		info[ix].buffer = { indexByteOffset,indexByteSize };
		for (ix=1u; ix<=propertyCount; ix++)
		{
			info[ix].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
			info[ix].buffer = { *(uploadByteOffsets++),*(uploadByteSizes++) };
		}
		for (ix=propertyCount+1u; ix<=propertyCount*2u; ix++)
		{
			info[ix].desc = core::smart_refctd_ptr<asset::IDescriptor>(upBuff);
			info[ix].buffer = { *(uploadByteOffsets++),*(uploadByteSizes++) };
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
	deferredReclaims.addEvent(GPUEventWrapper(std::move(fence)),DeferredDescriptorSetReclaimer(this,std::move(set)));
}