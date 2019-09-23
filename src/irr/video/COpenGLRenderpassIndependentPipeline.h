#ifndef __IRR_C_OPENGL_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_C_OPENGL_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/video/IGPURenderpassIndependentPipeline.h"
#include "COpenGLExtensionHandler.h"
#include "COpenGLSpecializedShader.h"

namespace irr {
namespace video
{

class COpenGLRenderpassIndependentPipeline : public IGPURenderpassIndependentPipeline
{
public:
    using IGPURenderpassIndependentPipeline::IGPURenderpassIndependentPipeline;

    struct SVAOHash
    {
        constexpr static size_t getHashLength()
        {
            return sizeof(hashVal)/sizeof(uint32_t);
        }

        bool operator!=(const SVAOHash& rhs) const 
        {
            for (size_t i = 0u; i < getHashLength(); ++i)
                if (hashVal[i] != rhs.hashVal[i])
                    return true;
            return false;
        }
        bool operator==(const SVAOHash& rhs) const
        {
            return !((*this) != rhs);
        }

        uint32_t getRelativeOffsetForAttrib(uint32_t _attr) const
        {
            if (_attr == 5u)
            {
                uint32_t val = (relOffsets[0] & (0xfu<<60)) >> 60;
                val |= (relOffsets[1] & 0xffu) << 4;
                return val;
            }
            if (_attr == 11u)
            {
                uint32_t val = (relOffsets[1] & (0xffu<<56)) >> 56;
                val |= (relOffsets[2] & 0xfu) << 8;
                return val;
            }
            const uint32_t ix = (_attr > 5u) + (_attr > 11u);
            const uint32_t shift = (_attr*12u) - (ix*64u);

            return (relOffsets[ix]>>shift) & 0xfffu;
        }

        union {
#include "irr/irrpack.h"
            struct {
                //can't find any info about what's guaranteed "min max" for rel offset, but here i'm assuming 2048 (same as with stride)
                uint64_t relOffsets[3];
                //GL's guaranteed minimal for max stride is 2048 so 12 bits per attrib needed
                //TODO do we want to include stride in hash? Stride is parameter of binding, while relative offset is parameter of attribute
                uint64_t strides[3];//16*12 bits
                uint64_t mapAttrToBinding;//16*4 bits
                uint16_t enabledAttribs;
                uint16_t divisors;
                uint8_t attribFormatAndComponentCount[16];
            } PACK_STRUCT;
#include "irr/irrunpack.h"
            uint32_t hashVal[7];
        };
    };

private:
    GLuint createGLobject(uint32_t _ctxID)
    {
        static_assert(SHADER_STAGE_COUNT == 5u, "SHADER_STAGE_COUNT is expected to be 5");
        const GLenum stages[SHADER_STAGE_COUNT]{ GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
        const GLenum stageFlags[SHADER_STAGE_COUNT]{ GL_VERTEX_SHADER_BIT, GL_TESS_CONTROL_SHADER_BIT, GL_TESS_EVALUATION_SHADER_BIT, GL_GEOMETRY_SHADER_BIT, GL_FRAGMENT_SHADER_BIT };

        GLuint pipeline = 0u;
        COpenGLExtensionHandler::extGlCreateProgramPipelines(1u, &pipeline);

        for (uint32_t ix = 0u; ix < SHADER_STAGE_COUNT; ++ix) {
            COpenGLSpecializedShader* glshdr = static_cast<COpenGLSpecializedShader*>(m_shaders[ix].get());
            GLuint progName = 0u;

            if (!glshdr || glshdr->getStage() != stages[ix])
                continue;
            progName = glshdr->getGLnameForCtx(_ctxID);

            if (progName)
                COpenGLExtensionHandler::extGlUseProgramStages(pipeline, stageFlags[ix], progName);
        }
        
        return pipeline;
    }
};

}}

#endif
