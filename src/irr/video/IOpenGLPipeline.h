#ifndef __IRR_I_OPENGL_PIPELINE_H_INCLUDED__
#define __IRR_I_OPENGL_PIPELINE_H_INCLUDED__

#include "irr/video/COpenGLSpecializedShader.h"
#include "irr/video/IGPUMeshBuffer.h"//for IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE

namespace irr { 
namespace video
{

template<size_t _STAGE_COUNT>
class IOpenGLPipeline
{
public:
    IOpenGLPipeline(uint32_t _ctxCount, uint32_t _ctxID, const GLuint _GLnames[_STAGE_COUNT], const COpenGLSpecializedShader::SProgramBinary _binaries[_STAGE_COUNT]) : 
        m_GLprograms(core::make_refctd_dynamic_array<decltype(m_GLprograms)>(_ctxCount*_STAGE_COUNT))
    {
        for (uint32_t i = 0u; i < _STAGE_COUNT; ++i)
            (*m_GLprograms)[i].GLname = _GLnames[i];
        for (uint32_t i = 1u; i < _ctxCount; ++i)
            for (uint32_t j = 0u; j < _STAGE_COUNT; ++j)
            {
                const auto& bin = _binaries[j];
                if (!bin.binary)
                    continue;
                const GLuint GLname = COpenGLExtensionHandler::extGlCreateProgram();
                COpenGLExtensionHandler::extGlProgramBinary(GLname, bin.format, bin.binary->data(), bin.binary->size());
                (*m_GLprograms)[i*_STAGE_COUNT+j].GLname = GLname;
            }

        const size_t uVals_sz = _STAGE_COUNT*_ctxCount*IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE;
        m_uniformValues = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(uVals_sz, 128));
    }
    ~IOpenGLPipeline()
    {
        //shader programs can be shared so all names can be freed by any thread
        for (const auto& p : (*m_GLprograms))
            if (p.GLname != 0u)
                COpenGLExtensionHandler::extGlDeleteProgram(p.GLname);
        _IRR_ALIGNED_FREE(m_uniformValues);
    }

    bool haveUniformsBeenEverSet(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t ix = _ctxID*_STAGE_COUNT + _stageIx;
        return !(*m_GLprograms)[ix].uniformsSetForTheVeryFirstTime;
    }
    void afterUniformsSet(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t ix = _ctxID*_STAGE_COUNT + _stageIx;
        (*m_GLprograms)[ix].uniformsSetForTheVeryFirstTime = false;
    }

    uint8_t* getPushConstantsStateForStage(uint32_t _stageIx, uint32_t _ctxID) const { return const_cast<uint8_t*>(m_uniformValues + ((_STAGE_COUNT*_ctxID + _stageIx)*IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)); }

protected:
    GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t name_ix = _ctxID*_STAGE_COUNT + _stageIx;

        return (*m_GLprograms)[name_ix].GLname;
    }

    struct SShaderProgram {
        GLuint GLname = 0u;
        bool uniformsSetForTheVeryFirstTime = true;
    };
    //mutable for deferred GL objects creation
    mutable core::smart_refctd_dynamic_array<SShaderProgram> m_GLprograms;
    uint8_t* m_uniformValues;
};

}}

#endif