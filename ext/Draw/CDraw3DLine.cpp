#include "../ext/Draw/CDraw3DLine.h"
#include "../ext/Draw/Draw3DLineShaders.h"

using namespace irr;
using namespace video;
using namespace scene;
using namespace ext;
using namespace draw;

const std::uint16_t CDraw3DLine::m_indices[2] = { 0, 1 };

uint32_t CDraw3DLine::offsets[2] =
{
    video::StreamingTransientDataBufferMT<>::invalid_address,
    video::StreamingTransientDataBufferMT<>::invalid_address
};

const uint32_t CDraw3DLine::alignments[2] =
{
    sizeof(decltype(S3DLineVertex::Position[0])),
    sizeof(std::uint16_t)
};

const uint32_t CDraw3DLine::sizes[2] =
{ sizeof(S3DLineVertex::Position), sizeof(std::uint16_t) };

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
        "","","",
        Draw3DLineFragmentShader,
        3,EMT_SOLID,
        callBack,
        0);
    callBack->drop();

    m_meshBuffer->setPrimitiveType(EPT_LINES);
    m_meshBuffer->setMeshDataAndFormat(m_desc);
    m_meshBuffer->setIndexBufferOffset(offsets[1]);
    m_meshBuffer->setIndexType(EIT_16BIT);
    m_meshBuffer->setIndexCount(2);
    m_lineData[2] = (void*) m_indices;
}

void CDraw3DLine::draw(
    float fromX, float fromY, float fromZ,
    float toX, float toY, float toZ,
    std::uint32_t r, std::uint32_t g, std::uint32_t b, std::uint32_t a)
{
    S3DLineVertex vertices[2] = {
        {{ fromX, fromY, fromZ }, { r, g, b, a }},
        {{ toX, toY, toZ }, { r, g, b, a }}
    };

    auto upStreamBuff = m_driver->getDefaultUpStreamingBuffer();
    m_lineData[0] = (void*) vertices;

    upStreamBuff->multi_place(2u, (const void* const*)m_lineData, (uint32_t*)&offsets,(uint32_t*)&sizes,(uint32_t*)&alignments, 0);
    if (upStreamBuff->needsManualFlushOrInvalidate())
    {
        auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
        m_driver->flushMappedMemoryRanges({{ upStreamMem,offsets[0],sizes[0] }, { upStreamMem,offsets[1],sizes[1] }});
    }

    auto buff = upStreamBuff->getBuffer();
    m_desc->mapVertexAttrBuffer(buff,EVAI_ATTR0,ECPA_THREE,ECT_FLOAT,sizeof(S3DLineVertex), offsetof(S3DLineVertex, Position[0]) + offsets[0]);
    m_desc->mapVertexAttrBuffer(buff,EVAI_ATTR1,ECPA_FOUR,ECT_FLOAT,sizeof(S3DLineVertex), offsetof(S3DLineVertex, Color[0]) + offsets[0]);
    m_desc->mapIndexBuffer(buff);

    m_driver->setTransform(E4X3TS_WORLD, core::matrix4x3());
    m_driver->setMaterial(m_material);
    m_driver->drawMeshBuffer(m_meshBuffer);

    upStreamBuff->multi_free(1u,(uint32_t*)&offsets,(uint32_t*)&sizes,m_driver->placeFence());
}

CDraw3DLine::~CDraw3DLine()
{
    m_desc->drop();
    m_meshBuffer->drop();
}
