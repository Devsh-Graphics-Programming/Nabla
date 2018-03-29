#include "IrrCompileConfig.h"
#include "COpenGLVAO.h"


#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace std
{
    size_t hash<irr::video::COpenGLVAOSpec::HashAttribs>::operator()(const irr::video::COpenGLVAOSpec::HashAttribs &x ) const
    {
        size_t retval = hash<uint64_t>()(x.hashVal[0]);

        for (size_t i=1; i<irr::video::COpenGLVAOSpec::HashAttribs::getHashLength(); i++)
            retval ^= hash<uint64_t>()(x.hashVal[i]);

        return retval;
    }
}

namespace irr
{
namespace video
{

COpenGLVAOSpec::COpenGLVAOSpec(core::LeakDebugger* dbgr) :  leakDebugger(dbgr)
{
    if (leakDebugger)
        leakDebugger->registerObj(this);

    for (size_t i=0; i<scene::EVAI_COUNT; i++)
        individualHashFields.setAttrFmtAndCompCnt(static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(i),compntsPerAttr[i],attrType[i]);
}

COpenGLVAOSpec::~COpenGLVAOSpec()
{
    if (leakDebugger)
        leakDebugger->deregisterObj(this);
}


void COpenGLVAOSpec::mapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, scene::E_COMPONENTS_PER_ATTRIBUTE components, scene::E_COMPONENT_TYPE type, const size_t &stride, size_t offset, uint32_t divisor)
{
    if (attrId>=scene::EVAI_COUNT)
#ifdef _DEBUG
    {
        os::Printer::log("MeshBuffer mapVertexAttrBuffer attribute ID out of range!\n",ELL_ERROR);
        return;
    }

    if (!validCombination(type,components))
    {
        os::Printer::log("MeshBuffer mapVertexAttrBuffer INVALID COMBINATION OF COMPONENT TYPE AND COUNT!\n",ELL_ERROR);
        return;
    }
#else
        return;
#endif // _DEBUG

    uint16_t mask = 0x1u<<attrId;
    uint16_t invMask = ~mask;


    size_t newStride;

    if (attrBuf)
    {
        attrBuf->grab();
        newStride = stride!=0 ? stride:scene::vertexAttrSize[type][components];
        //bind new buffer
        if (mappedAttrBuf[attrId])
            mappedAttrBuf[attrId]->drop();
        else
            individualHashFields.enabledAttribs |= mask;
    }
    else
    {
        if (mappedAttrBuf[attrId])
        {
            individualHashFields.enabledAttribs &= invMask;
            mappedAttrBuf[attrId]->drop();
        }
        components = scene::ECPA_FOUR;
        type = scene::ECT_FLOAT;
        newStride = 16;
        offset = 0;
        divisor = 0;
    }


    //set format
    if (components!=compntsPerAttr[attrId]||type!=attrType[attrId])
        individualHashFields.setAttrFmtAndCompCnt(attrId,components,type);


    const uint32_t maxDivisor = 0x1u<<_IRR_VAO_MAX_ATTRIB_DIVISOR_BITS;
    if (divisor>maxDivisor)
        divisor = maxDivisor;

    if (divisor!=attrDivisor[attrId])
    {
        for (size_t i=0; i<_IRR_VAO_MAX_ATTRIB_DIVISOR_BITS; i++)
        {
            if (divisor&(0x1u<<i))
                individualHashFields.attributeDivisors[i] |= mask; //set
            else
                individualHashFields.attributeDivisors[i] &= invMask; //zero out
        }

        attrDivisor[attrId] = divisor;
    }

    compntsPerAttr[attrId] = components;
    attrType[attrId] = type;
    attrStride[attrId] = newStride;
    attrOffset[attrId] = offset;


    mappedAttrBuf[attrId] = attrBuf;
}


}
}


#endif // _IRR_COMPILE_WITH_OPENGL_
