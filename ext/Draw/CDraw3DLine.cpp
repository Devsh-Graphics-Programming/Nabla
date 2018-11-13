#include "../ext/Draw/CDraw3DLine.h"
#include "../ext/Draw/Draw3DLineShaders.h"

using namespace irr;
using namespace video;
using namespace scene;
using namespace ext;
using namespace draw;

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
    m_meshBuffer->setMeshDataAndFormat(m_desc);
    m_meshBuffer->setIndexType(EIT_UNKNOWN);
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

    upStreamBuff->multi_place(2u, (const void* const*)lineData, (uint32_t*)&m_offsets,(uint32_t*)&sizes,(uint32_t*)&alignments);
    if (upStreamBuff->needsManualFlushOrInvalidate())
    {
        auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
        m_driver->flushMappedMemoryRanges({{ upStreamMem,m_offsets[0],sizes[0] }, { upStreamMem,m_offsets[1],sizes[1] }});
    }

    auto buff = upStreamBuff->getBuffer();
    m_desc->mapVertexAttrBuffer(buff,EVAI_ATTR0,ECPA_THREE,ECT_FLOAT,sizeof(S3DLineVertex), offsetof(S3DLineVertex, Position[0]) + m_offsets[0]);
    m_desc->mapVertexAttrBuffer(buff,EVAI_ATTR1,ECPA_FOUR,ECT_FLOAT,sizeof(S3DLineVertex), offsetof(S3DLineVertex, Color[0]) + m_offsets[0]);

    m_driver->setTransform(E4X3TS_WORLD, core::matrix4x3());
    m_driver->setMaterial(m_material);
    m_driver->drawMeshBuffer(m_meshBuffer);

    upStreamBuff->multi_free(2u,(uint32_t*)&m_offsets,(uint32_t*)&sizes,m_driver->placeFence());
}

CDraw3DLine::~CDraw3DLine()
{
    m_desc->drop();
    m_meshBuffer->drop();
}
