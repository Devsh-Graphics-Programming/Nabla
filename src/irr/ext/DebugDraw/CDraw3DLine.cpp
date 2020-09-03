#include "../ext/DebugDraw/CDraw3DLine.h"
#include "../ext/DebugDraw/Draw3DLineShaders.h"

using namespace irr;
using namespace video;
using namespace scene;
using namespace asset;
using namespace ext;
using namespace DebugDraw;

core::smart_refctd_ptr<CDraw3DLine> CDraw3DLine::create(IVideoDriver* _driver)
{
    return core::smart_refctd_ptr<CDraw3DLine>(new CDraw3DLine(_driver),core::dont_grab);
}

CDraw3DLine::CDraw3DLine(IVideoDriver* _driver) : m_driver(_driver), m_meshBuffer()
{
	auto vertexShader = m_driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(Draw3DLineVertexShader));
	auto fragShader = m_driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(Draw3DLineFragmentShader));

	auto vs = m_driver->createGPUSpecializedShader(vertexShader.get(),ISpecializedShader::SInfo({},nullptr,"main",ISpecializedShader::ESS_VERTEX));
	auto fs = m_driver->createGPUSpecializedShader(fragShader.get(),ISpecializedShader::SInfo({},nullptr,"main",ISpecializedShader::ESS_FRAGMENT));

	asset::SPushConstantRange pcRange[1] = {ISpecializedShader::ESS_VERTEX,0,sizeof(core::matrix4SIMD)};
	auto pLayout = m_driver->createGPUPipelineLayout(pcRange,pcRange+1u,nullptr,nullptr,nullptr,nullptr);


	IGPUSpecializedShader* shaders[2] = {vs.get(),fs.get()};

	SVertexInputParams inputParams;
	inputParams.enabledAttribFlags = 0b11u;
	inputParams.enabledBindingFlags = 0b1u;
	inputParams.attributes[0].binding = 0u;
	inputParams.attributes[0].format = EF_R32G32B32_SFLOAT;
	inputParams.attributes[0].relativeOffset = offsetof(S3DLineVertex, Position[0]);
	inputParams.attributes[1].binding = 0u;
	inputParams.attributes[1].format = EF_R32G32B32A32_SFLOAT;
	inputParams.attributes[1].relativeOffset = offsetof(S3DLineVertex, Color[0]);
	inputParams.bindings[0].stride = sizeof(S3DLineVertex);
	inputParams.bindings[0].inputRate = EVIR_PER_VERTEX;

	SBlendParams blendParams;
	blendParams.logicOpEnable = false;
	blendParams.logicOp = ELO_NO_OP;
	for (size_t i=1ull; i<SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
		blendParams.blendParams[i].attachmentEnabled = false;

	SPrimitiveAssemblyParams assemblyParams = {EPT_LINE_LIST,false,2u};

	SStencilOpParams defaultStencil;
	SRasterizationParams rasterParams;
	rasterParams.polygonMode = EPM_LINE;
	auto pipeline = m_driver->createGPURenderpassIndependentPipeline(	nullptr,std::move(pLayout),shaders,shaders+sizeof(shaders)/sizeof(void*),
																		inputParams,blendParams,assemblyParams,rasterParams);


	SBufferBinding<IGPUBuffer> bindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = {0u,core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->getDefaultUpStreamingBuffer()->getBuffer())};
	m_meshBuffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(std::move(pipeline),nullptr,bindings,SBufferBinding<IGPUBuffer>{});
	m_meshBuffer->setIndexType(EIT_UNKNOWN);
	m_meshBuffer->setIndexCount(2);
}

void CDraw3DLine::draw(const core::matrix4SIMD& viewProjMat,
    float fromX, float fromY, float fromZ,
    float toX, float toY, float toZ,
    float r, float g, float b, float a)
{
    S3DLineVertex vertices[2] = {
        {{ fromX, fromY, fromZ }, { r, g, b, a }},
        {{ toX, toY, toZ }, { r, g, b, a }}
    };

    auto upStreamBuff = m_driver->getDefaultUpStreamingBuffer();
    void* lineData[1] = { vertices };

    static const uint32_t sizes[1] = { sizeof(S3DLineVertex) * 2 };
    uint32_t offset[1] = { video::StreamingTransientDataBufferMT<>::invalid_address };
    upStreamBuff->multi_place(1u, (const void* const*)lineData, (uint32_t*)&offset,(uint32_t*)&sizes,(uint32_t*)&alignments);
    if (upStreamBuff->needsManualFlushOrInvalidate())
    {
        auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
        m_driver->flushMappedMemoryRanges({{ upStreamMem,offset[0],sizes[0] }});
    }

    m_meshBuffer->setBaseVertex(offset[0]/sizeof(S3DLineVertex));


	m_driver->bindGraphicsPipeline(m_meshBuffer->getPipeline());
	m_driver->pushConstants(m_meshBuffer->getPipeline()->getLayout(),ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD),viewProjMat.pointer());

    m_driver->drawMeshBuffer(m_meshBuffer.get());

    upStreamBuff->multi_free(1u,(uint32_t*)&offset,(uint32_t*)&sizes,std::move(m_driver->placeFence()));
}

void CDraw3DLine::draw(const core::matrix4SIMD& viewProjMat, const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
{
    auto upStreamBuff = m_driver->getDefaultUpStreamingBuffer();
    const void* lineData[1] = { linesData.data() };

    const uint32_t sizes[1] = { sizeof(S3DLineVertex) * linesData.size() * 2 };
    uint32_t offset[1] = { video::StreamingTransientDataBufferMT<>::invalid_address };
    upStreamBuff->multi_place(1u, (const void* const*)lineData, (uint32_t*)&offset,(uint32_t*)&sizes,(uint32_t*)&alignments);
    if (upStreamBuff->needsManualFlushOrInvalidate())
    {
        auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
        m_driver->flushMappedMemoryRanges({{ upStreamMem,offset[0],sizes[0] }});
    }

    m_meshBuffer->setBaseVertex(offset[0]/sizeof(S3DLineVertex));
    m_meshBuffer->setIndexCount(linesData.size() * 2);
	
	m_driver->bindGraphicsPipeline(m_meshBuffer->getPipeline());
	m_driver->pushConstants(m_meshBuffer->getPipeline()->getLayout(),ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD),viewProjMat.pointer());

    m_driver->drawMeshBuffer(m_meshBuffer.get());

    upStreamBuff->multi_free(1u,(uint32_t*)&offset,(uint32_t*)&sizes,std::move(m_driver->placeFence()));
}