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
    //! _binaries' elements are getting move()'d!
    IOpenGLPipeline(uint32_t _ctxCount, uint32_t _ctxID, const GLuint _GLnames[_STAGE_COUNT], COpenGLSpecializedShader::SProgramBinary _binaries[_STAGE_COUNT]) :
        m_GLnames(core::make_refctd_dynamic_array<decltype(m_GLnames)>(_STAGE_COUNT*_ctxCount)),
        m_shaderBinaries(core::make_refctd_dynamic_array<decltype(m_shaderBinaries)>(_STAGE_COUNT)),
        m_uniformsSetForTheVeryFirstTime(core::make_refctd_dynamic_array<decltype(m_uniformsSetForTheVeryFirstTime)>(_STAGE_COUNT*_ctxCount))
    {
        memset(m_GLnames->data(), 0, m_GLnames->size()*sizeof(GLuint));
        memset(m_uniformsSetForTheVeryFirstTime->data(), 0, m_uniformsSetForTheVeryFirstTime->size()*sizeof(bool));
        memcpy(m_GLnames->data()+_ctxID*_STAGE_COUNT, _GLnames, _STAGE_COUNT*sizeof(GLuint));
        std::move(_binaries, _binaries+_STAGE_COUNT, m_shaderBinaries->begin());

        const size_t uVals_sz = _STAGE_COUNT*_ctxCount*IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE;
        m_uniformValues = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(uVals_sz, 128));
    }
    ~IOpenGLPipeline()
    {
        //shader programs can be shared so all names can be freed by any thread
        for (GLuint n : *m_GLnames)
            if (n != 0u)
                COpenGLExtensionHandler::extGlDeleteProgram(n);
        _IRR_ALIGNED_FREE(m_uniformValues);
    }

    bool haveUniformsBeenEverSet(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t ix = _ctxID*_STAGE_COUNT + _stageIx;
        return !(*m_uniformsSetForTheVeryFirstTime)[ix];
    }
    void afterUniformsSet(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t ix = _ctxID*_STAGE_COUNT + _stageIx;
        (*m_uniformsSetForTheVeryFirstTime)[ix] = false;
    }

    uint8_t* getPushConstantsStateForStage(uint32_t _stageIx, uint32_t _ctxID) const { return const_cast<uint8_t*>(m_uniformValues + ((_STAGE_COUNT*_ctxID + _stageIx)*IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)); }

protected:
    GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t name_ix = _ctxID*_STAGE_COUNT + _stageIx;

        if ((*m_GLnames)[name_ix])
            return (*m_GLnames)[name_ix];
        else if ((*m_shaderBinaries)[_stageIx].binary)
        {
            const auto& bin = (*m_shaderBinaries)[_stageIx];

            const GLuint GLname = COpenGLExtensionHandler::extGlCreateProgram();
            COpenGLExtensionHandler::extGlProgramBinary(GLname, bin.format, bin.binary->data(), bin.binary->size());
            (*m_GLnames)[_ctxID] = GLname;
            return GLname;
        }

        return (*m_GLnames)[name_ix];
    }

    //mutable for deferred GL objects creation
    mutable core::smart_refctd_dynamic_array<GLuint> m_GLnames;
    mutable core::smart_refctd_dynamic_array<COpenGLSpecializedShader::SProgramBinary> m_shaderBinaries;
    mutable core::smart_refctd_dynamic_array<bool> m_uniformsSetForTheVeryFirstTime;
    uint8_t* m_uniformValues;
};

}}

#endif