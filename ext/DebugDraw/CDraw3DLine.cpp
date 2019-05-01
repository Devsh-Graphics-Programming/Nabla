#include "../ext/DebugDraw/CDraw3DLine.h"
#include "../ext/DebugDraw/Draw3DLineShaders.h"

using namespace irr;
using namespace video;
using namespace scene;
using namespace asset;
using namespace ext;
using namespace DebugDraw;

CDraw3DLine* CDraw3DLine::create(IVideoDriver* _driver)
{
    return new CDraw3DLine(_driver);
}

CDraw3DLine::CDraw3DLine(IVideoDriver* _driver)
:   m_driver(_driver),
    m_desc(_driver->createGPUMeshDataFormatDesc()),
    m_meshBuffer(new IGPUMeshBuffer())
{
    auto callBack = new Draw3DLineCallBack();
    m_material.MaterialType = (E_MATERIAL_TYPE)
        m_driver->getGPUProgrammingServices()->addHighLevelShaderMaterial(
        Draw3DLineVertexShader,
        nullptr,nullptr,nullptr,
        Draw3DLineFragmentShader,
        2,EMT_SOLID,
        callBack,
        0);
    callBack->drop();

    m_meshBuffer->setPrimitiveType(EPT_LINES);
    m_meshBuffer->setIndexType(EIT_UNKNOWN);
    m_meshBuffer->setIndexCount(2);

    auto buff = m_driver->getDefaultUpStreamingBuffer()->getBuffer();
    m_desc->setVertexAttrBuffer(buff,EVAI_ATTR0,EF_R32G32B32_SFLOAT,sizeof(S3DLineVertex), offsetof(S3DLineVertex, Position[0]));
    m_desc->setVertexAttrBuffer(buff,EVAI_ATTR1,EF_R32G32B32A32_SFLOAT,sizeof(S3DLineVertex), offsetof(S3DLineVertex, Color[0]));
    m_meshBuffer->setMeshDataAndFormat(m_desc);
    m_desc->drop();
}

void CDraw3DLine::draw(
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

    m_driver->setTransform(E4X3TS_WORLD, core::matrix4x3());
    m_driver->setMaterial(m_material);
    m_driver->drawMeshBuffer(m_meshBuffer);

    upStreamBuff->multi_free(1u,(uint32_t*)&offset,(uint32_t*)&sizes,std::move(m_driver->placeFence()));
}

void CDraw3DLine::draw(const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
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

    m_driver->setTransform(E4X3TS_WORLD, core::matrix4x3());
    m_driver->setMaterial(m_material);
    m_driver->drawMeshBuffer(m_meshBuffer);

    upStreamBuff->multi_free(1u,(uint32_t*)&offset,(uint32_t*)&sizes,std::move(m_driver->placeFence()));
}

CDraw3DLine::~CDraw3DLine()
{
    m_meshBuffer->drop();
}
