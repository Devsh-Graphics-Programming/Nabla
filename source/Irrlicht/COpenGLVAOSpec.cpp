#include "IrrCompileConfig.h"
#include "COpenGLVAOSpec.h"


#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace std
{
	template <>
	struct hash<irr::video::COpenGLVAOSpec::HashAttribs>
	{
		std::size_t operator()(const irr::video::COpenGLVAOSpec::HashAttribs& x) const noexcept
		{
			size_t retval = hash<uint64_t>()(x.hashVal[0]);

			for (size_t i = 1; i<irr::video::COpenGLVAOSpec::HashAttribs::getHashLength(); i++)
				retval ^= hash<uint64_t>()(x.hashVal[i]);

			return retval;
		}
	};
}

namespace irr
{
namespace video
{

COpenGLVAOSpec::COpenGLVAOSpec(core::LeakDebugger* dbgr) :  leakDebugger(dbgr)
{
    if (leakDebugger)
        leakDebugger->registerObj(this);

    for (size_t i=0; i<asset::EVAI_COUNT; i++)
        individualHashFields.setAttrFmt(static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(i),attrFormat[i]);
}

COpenGLVAOSpec::~COpenGLVAOSpec()
{
    if (leakDebugger)
        leakDebugger->deregisterObj(this);
}


void COpenGLVAOSpec::setVertexAttrBuffer(IGPUBuffer* attrBuf, asset::E_VERTEX_ATTRIBUTE_ID attrId, asset::E_FORMAT format, size_t stride, size_t offset, uint32_t divisor)
{
    if (attrId>= asset::EVAI_COUNT)
#ifdef _IRR_DEBUG
    {
        os::Printer::log("MeshBuffer setVertexAttrBuffer attribute ID out of range!\n",ELL_ERROR);
        return;
    }
#else
        return;
#endif // _IRR_DEBUG

    uint16_t mask = 0x1u<<attrId;
    uint16_t invMask = ~mask;


    size_t newStride;

    if (attrBuf)
    {
        attrBuf->grab();
        newStride = stride!=0u ? stride : getTexelOrBlockSize(format);
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
        format = asset::EF_R32G32B32A32_SFLOAT;
        newStride = 16u;
        offset = 0u;
        divisor = 0u;
    }


    individualHashFields.setAttrFmt(attrId, format);


    const uint32_t maxDivisor = 1u;
    if (divisor>maxDivisor)
        divisor = maxDivisor;

    if (divisor!=getAttribDivisor(attrId))
    {
        if (divisor)
            attrDivisor |= (divisor<<attrId);
        else
            attrDivisor &= ~(divisor<<attrId);
        individualHashFields.attributeDivisors = attrDivisor;
    }

    attrFormat[attrId] = format;
    attrStride[attrId] = newStride;
    attrOffset[attrId] = offset;


    mappedAttrBuf[attrId] = attrBuf;
}


}
}


#endif // _IRR_COMPILE_WITH_OPENGL_
