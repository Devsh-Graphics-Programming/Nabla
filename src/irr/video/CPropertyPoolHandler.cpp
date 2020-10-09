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
	const auto maxSSBO = m_driver->getMaxSSBOBindings(); // TODO: make sure non dynamic
	assert(maxSSBO>MaxPropertiesPerCS);
	m_pipelineCount = (maxSSBO-1u)/2u;
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

		auto shader = m_driver->createGPUShader(std::move(cpushader));
		auto specshader = m_driver->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});

		IGPUDescriptorSetLayout::SBinding bindings[3];
		for (auto j=0; j<3; j++)
		{
			bindings[j].binding = j;
			bindings[j].type = asset::EDT_STORAGE_BUFFER;
			bindings[j].count = j ? propCount:1u;
			bindings[j].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
			bindings[j].samplers = nullptr;
		}
		m_descriptorSetLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+3);

		auto layout = m_driver->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(m_descriptorSetLayout));

		assert(!m_pipelines[i]); // protect against compiler shit
		m_pipelines[i] = m_driver->createGPUComputePipeline(pipelineCache,std::move(layout),std::move(specshader));
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