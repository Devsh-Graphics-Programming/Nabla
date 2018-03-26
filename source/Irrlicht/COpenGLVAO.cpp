#include "IrrCompileConfig.h"
#include "COpenGLVAO.h"
#include "COpenGLExtensionHandler.h"


#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

COpenGLVAO::COpenGLVAO(core::LeakDebugger* dbgr) :  vao(0), lastValidated(0), leakDebugger(dbgr)
{
    if (leakDebugger)
        leakDebugger->registerObj(this);

    COpenGLExtensionHandler::extGlCreateVertexArrays(1,&vao);
}

COpenGLVAO::~COpenGLVAO()
{
    if (leakDebugger)
        leakDebugger->deregisterObj(this);

    if (vao)
        COpenGLExtensionHandler::extGlDeleteVertexArrays(1,&vao);
}


void COpenGLVAO::mapIndexBuffer(IGPUBuffer* ixbuf)
{
    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(ixbuf);
    if (ixbuf)
    {
        asGLBuf->grab();
        if (asGLBuf!=mappedIndexBuf)
        {
            COpenGLExtensionHandler::extGlVertexArrayElementBuffer(vao,asGLBuf->getOpenGLName());
            individualHashFields.elementArrayBinding = asGLBuf->getOpenGLName();
        }
        if (mappedIndexBuf)
            mappedIndexBuf->drop();
    }
    else
    {
        if (mappedIndexBuf)
        {
            COpenGLExtensionHandler::extGlVertexArrayElementBuffer(vao,0);
            individualHashFields.elementArrayBinding = 0;
            mappedIndexBuf->drop();
        }
    }

    mappedIndexBuf = asGLBuf;
}


void COpenGLVAO::mapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, scene::E_COMPONENTS_PER_ATTRIBUTE components, scene::E_COMPONENT_TYPE type, const size_t &stride, size_t offset, uint32_t divisor)
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


    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(attrBuf);
    size_t newStride;

    if (attrBuf)
    {
        asGLBuf->grab();
        newStride = stride!=0 ? stride:scene::vertexAttrSize[type][components];
        //bind new buffer
        if (mappedAttrBuf[attrId])
        {
            mappedAttrBuf[attrId]->drop();
            if (newStride!=attrStride[attrId]||offset!=attrOffset[attrId]||asGLBuf!=mappedAttrBuf[attrId]) //dont compare openGL names, could have been recycled
                COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,attrId,asGLBuf->getOpenGLName(),offset,newStride); //kill
        }
        else
        {
            individualHashFields.enabledAttribs |= 0x1u<<attrId;
            COpenGLExtensionHandler::extGlEnableVertexArrayAttrib(vao,attrId);
            COpenGLExtensionHandler::extGlVertexArrayAttribBinding(vao,attrId,attrId);
            COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,attrId,asGLBuf->getOpenGLName(),offset,newStride); //kill
        }

        //! maybe only set format if we're actually going to use the buffer???
    }
    else
    {
        if (mappedAttrBuf[attrId])
        {
            COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,attrId,0,0,16); //kill
            COpenGLExtensionHandler::extGlDisableVertexArrayAttrib(vao,attrId);
            individualHashFields.enabledAttribs &= ~(0x1u<<attrId);
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
    {
        switch (type)
        {
            case scene::ECT_FLOAT:
            case scene::ECT_HALF_FLOAT:
            case scene::ECT_DOUBLE_IN_FLOAT_OUT:
            case scene::ECT_UNSIGNED_INT_10F_11F_11F_REV:
            //INTEGER FORMS
            case scene::ECT_NORMALIZED_INT_2_10_10_10_REV:
            case scene::ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV:
            case scene::ECT_NORMALIZED_BYTE:
            case scene::ECT_NORMALIZED_UNSIGNED_BYTE:
            case scene::ECT_NORMALIZED_SHORT:
            case scene::ECT_NORMALIZED_UNSIGNED_SHORT:
            case scene::ECT_NORMALIZED_INT:
            case scene::ECT_NORMALIZED_UNSIGNED_INT:
            case scene::ECT_INT_2_10_10_10_REV:
            case scene::ECT_UNSIGNED_INT_2_10_10_10_REV:
            case scene::ECT_BYTE:
            case scene::ECT_UNSIGNED_BYTE:
            case scene::ECT_SHORT:
            case scene::ECT_UNSIGNED_SHORT:
            case scene::ECT_INT:
            case scene::ECT_UNSIGNED_INT:
                COpenGLExtensionHandler::extGlVertexArrayAttribFormat(vao,attrId,eComponentsPerAttributeToGLint[components],eComponentTypeToGLenum[type],scene::isNormalized(type) ? GL_TRUE:GL_FALSE,0);
                individualHashFields.setAttrFmtAndCompCnt(attrId,components,type);
                break;
            case scene::ECT_INTEGER_INT_2_10_10_10_REV:
            case scene::ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV:
            case scene::ECT_INTEGER_BYTE:
            case scene::ECT_INTEGER_UNSIGNED_BYTE:
            case scene::ECT_INTEGER_SHORT:
            case scene::ECT_INTEGER_UNSIGNED_SHORT:
            case scene::ECT_INTEGER_INT:
            case scene::ECT_INTEGER_UNSIGNED_INT:
                COpenGLExtensionHandler::extGlVertexArrayAttribIFormat(vao,attrId,eComponentsPerAttributeToGLint[components],eComponentTypeToGLenum[type],0);
                individualHashFields.setAttrFmtAndCompCnt(attrId,components,type);
                break;
        //special
            case scene::ECT_DOUBLE_IN_DOUBLE_OUT:
                COpenGLExtensionHandler::extGlVertexArrayAttribLFormat(vao,attrId,eComponentsPerAttributeToGLint[components],GL_DOUBLE,0);
                individualHashFields.setAttrFmtAndCompCnt(attrId,components,type);
                break;
        }
    }

    const uint32_t maxDivisor = 0x1u<<_IRR_VAO_MAX_ATTRIB_DIVISOR_BITS;
    if (divisor>maxDivisor)
        divisor = maxDivisor;

    if (divisor!=attrDivisor[attrId])
    {
        COpenGLExtensionHandler::extGlVertexArrayBindingDivisor(vao,attrId,divisor);
        for (size_t i=0; i<_IRR_VAO_MAX_ATTRIB_DIVISOR_BITS; i++)
        {
            uint16_t mask = (0x1u<<attrId);
            uint16_t invMask = ~mask;

            individualHashFields.attributeDivisors[i] &= invMask; //zero out
            individualHashFields.attributeDivisors[i] |= divisor&mask; //then possibly set
        }

        attrDivisor[attrId] = divisor;
    }

    compntsPerAttr[attrId] = components;
    attrType[attrId] = type;
    attrStride[attrId] = newStride;
    attrOffset[attrId] = offset;


    mappedAttrBuf[attrId] = asGLBuf;
}

void COpenGLVAO::setMappedBufferOffset(const scene::E_VERTEX_ATTRIBUTE_ID& attrId, const size_t &offset)
{
    if (attrId>=scene::EVAI_COUNT)
#ifdef _DEBUG
    {
        os::Printer::log("MeshBuffer mapVertexAttrBuffer attribute ID out of range!\n",ELL_ERROR);
        return;
    }
#else
        return;
#endif // _DEBUG

    COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(mappedAttrBuf[attrId]);
    if (!asGLBuf)
        return;


    if (offset!=attrOffset[attrId]) //dont compare openGL names, could have been recycled
        COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,attrId,asGLBuf->getOpenGLName(),offset,attrStride[attrId]); //kill

    attrOffset[attrId] = offset;
}


bool COpenGLVAO::rebindRevalidate()
{
    uint64_t highestRevalidateStamp = lastValidated;

    if (mappedIndexBuf)
    {
        uint64_t revalidateStamp = mappedIndexBuf->getLastTimeReallocated();
        if (revalidateStamp>lastValidated)
        {
            highestRevalidateStamp = revalidateStamp;

            COpenGLBuffer* asGLBuf = static_cast<COpenGLBuffer*>(mappedIndexBuf);
            COpenGLExtensionHandler::extGlVertexArrayElementBuffer(vao,asGLBuf->getOpenGLName());
            individualHashFields.elementArrayBinding = asGLBuf->getOpenGLName();
        }
    }

    for (size_t i=0; i<scene::EVAI_COUNT; i++)
    {
        if (!mappedAttrBuf[i])
            continue;

        COpenGLBuffer* buffer = static_cast<COpenGLBuffer*>(mappedAttrBuf[i]);

        uint64_t revalidateStamp = buffer->getLastTimeReallocated();
        if (revalidateStamp>lastValidated)
        {
            if (revalidateStamp>highestRevalidateStamp)
                highestRevalidateStamp = revalidateStamp;
            COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,i,buffer->getOpenGLName(),attrOffset[i],attrStride[i]); //kill
        }
    }

    lastValidated = highestRevalidateStamp;

	return true;
}


}
}


#endif // _IRR_COMPILE_WITH_OPENGL_
