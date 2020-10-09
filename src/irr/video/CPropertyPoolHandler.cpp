#include "irr/video/CPropertyPoolHandler.h"
#include "irr/video/CPropertyPool.h"

using namespace irr;
using namespace video;

//
constexpr char* copyCsSource = R"(
layout(local_size_x=_IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_) in;

layout(set=0,binding=0) readonly restrict buffer Indices
{
    uint elementCount;
	int propertyDWORDsize_upDownFlag[_IRR_BUILTIN_PROPERTY_COUNT_];
    uint indices[];
};


layout(set=1, binding=0) readonly restrict buffer InData
{
    uint data[];
} inBuff[_IRR_BUILTIN_PROPERTY_COUNT_];
layout(set=1, binding=1) writeonly restrict buffer OutData
{
    uint data[];
} outBuff[_IRR_BUILTIN_PROPERTY_COUNT_];


uint shared workgroupShared[_IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_];


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
		workgroupShared[localIx] = indices[localIx];
	barrier();
	memoryBarrier();
#endif
    const uint index = gl_GlobalInvocationID.x/propDWORDs;
    if (index>=elementCount)
        return;

	const uint redir = (
#if 0 //optimization
		workgroupShared[]
#else 
		indices[index
#endif
	// its equivalent to `indices[index]*propDWORDs+gl_GlobalInvocationID.x%propDWORDs`
    -index)*propDWORDs+gl_GlobalInvocationID.x;

    const uint inIndex = download ? redir:gl_GlobalInvocationID.x;
    const uint outIndex = download ? gl_GlobalInvocationID.x:redir;
	outBuff[propID].data[outIndex] = inBuff[propID].data[inIndex];
}
)";

//
CPropertyPoolHandler::CPropertyPoolHandler(IVideoDriver* driver) : m_driver(driver), m_copyPipelines{nullptr}
{
	assert(driver);
	m_pipelineCount = (driver->getMaxSSBOBindings()-1u)/2u;
	for (uint32_t i=0u; i<m_pipelineCount; i++)
	{
		const auto propCount = i+1u;

		std::string shaderSource("#version 440 core\n");
		// property count
		shaderSource += "#define _IRR_BUILTIN_PROPERTY_COUNT_ ";
		shaderSource += std::to_string(propCount)+"\n";
		// workgroup sizes
		shaderSource += "#define _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_ ";
		shaderSource += std::to_string(IdealWorkGroupSize)+"\n";
		//
		shaderSource += copyCsSource;

		auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSource.c_str());

		auto shader = driver->createGPUShader(std::move(cpushader));
		auto specshader = driver->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});

		IGPUDescriptorSetLayout::SBinding bindings[2];
		{
			bindings[0].
		}
		auto elementDSLayout = driver->createGPUDescriptorSetLayout(bindings,bindings+1);
		{
			bindings[0].
			bindings[1].
		}
		auto copyBuffersDSLayout = driver->createGPUDescriptorSetLayout(bindings,bindings+2);

		auto layout = driver->createGPUPipelineLayout(nullptr,nullptr,std::move(elementDSLayout),std::move(copyBuffersDSLayout));

		m_pipelines[i] = driver->createGPUComputePipeline(pipelineCache,std::move(layout),std::move(specshader));
	}
}

//
bool CPropertyPoolHandler::addProperties(const IPropertyPool* poolsBegin, const IPropertyPool* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd)
{
	bool success = true;

	auto poolIndicesBegin = indicesBegin;
	auto poolIndicesEnd = indicesEnd;
	for (auto it=poolsBegin; it!=poolsEnd; it++)
	{
		success = allocateProperties(outIndicesBegin,outIndicesEnd) && success;
		poolIndicesBegin++;
		poolIndicesEnd++;
	}

	return uploadProperties(poolsBegin,poolsEnd,outIndicesBegin,outIndicesEnd,dataBegin,dataEnd) && success;
}

//
bool CPropertyPoolHandler::uploadProperties(const IPropertyPool* poolsBegin, const IPropertyPool* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd);
{
}

//
bool CPropertyPoolHandler::downloadProperties(const IPropertyPool* poolsBegin, const IPropertyPool* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, void* const* const* dataBegin, void* const* const* dataEnd)
{
#if 0
	const auto passCount = getPipelineCount();
	core::smart_refctd_ptr<IGPUComputePipeline> passes[passCount];
	for (auto pass = 0u; pass < passCount; pass++)
	{
		//driver->bindComputePipeline();
		//driver->bindDescriptorSets(EPBP_COMPUTE,layout,0u,1u,&set,&offsets);
		//driver->dispatch(getWorkGroupSizeX(propertiesThisPass),propertiesThisPass,1u);
	}
#endif
	return false;
}