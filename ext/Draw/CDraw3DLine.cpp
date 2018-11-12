#include "../ext/Draw/CDraw3DLine.h"
#include "../ext/Draw/Draw3DLineShaders.h"

using namespace irr;
using namespace ext;
using namespace draw;

CDraw3DLine* CDraw3DLine::create(video::IVideoDriver* _driver)
{
    return new CDraw3DLine(_driver);
}

CDraw3DLine::CDraw3DLine(video::IVideoDriver* _driver)
:   m_driver(_driver),
    m_desc(_driver->createGPUMeshDataFormatDesc()),
    m_meshBuffer(new scene::IGPUMeshBuffer())
{
    auto callBack = new Draw3DLineCallBack();
    m_material.MaterialType = (video::E_MATERIAL_TYPE)
        m_driver->getGPUProgrammingServices()->addHighLevelShaderMaterial(
        Draw3DLineVertexShader,
        "","","",
        Draw3DLineFragmentShader,
        3,video::EMT_SOLID,
        callBack,
        0);
    callBack->drop();
}

void CDraw3DLine::draw(const S3DLine& line)
{
    auto upStreamBuff = m_driver->getDefaultUpStreamingBuffer();
    m_lineData[0] = (void*) line.Start;
    m_lineData[1] = (void*) line.End;
    m_lineData[2] = (void*) line.Color;

    upStreamBuff->multi_place(2u, (const void* const*)m_lineData,(uint32_t*)&offsets,(uint32_t*)&sizes,(uint32_t*)&alignments);
    if (upStreamBuff->needsManualFlushOrInvalidate())
    {
        auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
        m_driver->flushMappedMemoryRanges({video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[0],sizes[0]),video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[1],sizes[1])});
    }

    auto buff = upStreamBuff->getBuffer();
    m_desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,sizeof(S3DLine), offsetof(S3DLine, Start[0]) + offsets[0]);
    m_desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR1,scene::ECPA_THREE,scene::ECT_FLOAT,sizeof(S3DLine),offsetof(S3DLine, End[0]) + offsets[0]);
    m_desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR2,scene::ECPA_FOUR,scene::ECT_FLOAT,sizeof(S3DLine),offsetof(S3DLine, Color[0]) + offsets[0]);
    m_desc->mapIndexBuffer(buff);

    m_meshBuffer->setMeshDataAndFormat(m_desc);
    m_meshBuffer->setIndexCount(1);

    m_driver->setTransform(video::E4X3TS_WORLD, core::matrix4x3());
    m_driver->setMaterial(m_material);
    m_driver->drawMeshBuffer(m_meshBuffer);

    upStreamBuff->multi_free(1u,(uint32_t*)&offsets,(uint32_t*)&sizes,m_driver->placeFence());
}

CDraw3DLine::~CDraw3DLine()
{
    m_desc->drop();
    m_meshBuffer->drop();
}
