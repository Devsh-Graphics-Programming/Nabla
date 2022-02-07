// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_OPENGL_PIPELINE_H_INCLUDED__
#define __NBL_VIDEO_I_OPENGL_PIPELINE_H_INCLUDED__

#include "nbl/video/COpenGLSpecializedShader.h"
#include "nbl/video/IGPUMeshBuffer.h"  //for IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE

#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
{
namespace video
{
template<size_t _STAGE_COUNT>
class IOpenGLPipeline
{
protected:
    // needed for spirv-cross-based workaround of GL's behaviour of gl_InstanceID
    struct SBaseInstance
    {
        GLint cache = 0;
        GLint id = -1;
    };

private:
    using base_instance_cache_t = SBaseInstance;

    _NBL_STATIC_INLINE_CONSTEXPR bool IsComputePipelineBase = (_STAGE_COUNT == 1u);
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t BaseInstancePerContextCacheSize = IsComputePipelineBase ? 0ull : sizeof(base_instance_cache_t);
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t UniformsPerContextCacheSize = _STAGE_COUNT * IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE + BaseInstancePerContextCacheSize;

    static uint32_t baseInstanceCacheByteoffsetForCtx(uint32_t _ctxId)
    {
        return UniformsPerContextCacheSize * _ctxId;
    }
    static uint32_t uniformsCacheByteoffsetForCtx(uint32_t _ctxId)
    {
        return baseInstanceCacheByteoffsetForCtx(_ctxId) + BaseInstancePerContextCacheSize;
    }
    static uint32_t uniformsCacheByteoffsetForCtxAndStage(uint32_t _ctxId, uint32_t _stage)
    {
        return uniformsCacheByteoffsetForCtx(_ctxId) + _stage * IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE;
    }

public:
    IOpenGLPipeline(uint32_t _ctxCount, uint32_t _ctxID, const GLuint _GLnames[_STAGE_COUNT], const COpenGLSpecializedShader::SProgramBinary _binaries[_STAGE_COUNT])
        : m_GLprograms(core::make_refctd_dynamic_array<decltype(m_GLprograms)>(_ctxCount * _STAGE_COUNT))
    {
        for(uint32_t i = 0u; i < _STAGE_COUNT; ++i)
            (*m_GLprograms)[i].GLname = _GLnames[i];
        for(uint32_t i = 1u; i < _ctxCount; ++i)
            for(uint32_t j = 0u; j < _STAGE_COUNT; ++j)
            {
                const auto& bin = _binaries[j];
                if(!bin.binary)
                    continue;
                const GLuint GLname = COpenGLExtensionHandler::extGlCreateProgram();
                COpenGLExtensionHandler::extGlProgramBinary(GLname, bin.format, bin.binary->data(), static_cast<uint32_t>(bin.binary->size()));
                (*m_GLprograms)[i * _STAGE_COUNT + j].GLname = GLname;
            }

        const size_t uVals_sz = UniformsPerContextCacheSize * _ctxCount;
        m_uniformValues = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(uVals_sz, 128));
        for(uint32_t i = 0u; i < _ctxCount; ++i)
            getBaseInstanceState(i)[0] = base_instance_cache_t{};
    }
    ~IOpenGLPipeline()
    {
        //shader programs can be shared so all names can be freed by any thread
        for(const auto& p : (*m_GLprograms))
            if(p.GLname != 0u)
                COpenGLExtensionHandler::extGlDeleteProgram(p.GLname);
        _NBL_ALIGNED_FREE(m_uniformValues);
    }

    uint8_t* getPushConstantsStateForStage(uint32_t _stageIx, uint32_t _ctxID) const { return const_cast<uint8_t*>(m_uniformValues + uniformsCacheByteoffsetForCtxAndStage(_ctxID, _stageIx)); }
    base_instance_cache_t* getBaseInstanceState(uint32_t _ctxID) const { return const_cast<base_instance_cache_t*>(reinterpret_cast<const base_instance_cache_t*>(m_uniformValues + baseInstanceCacheByteoffsetForCtx(_ctxID))); }

protected:
    void setUniformsImitatingPushConstants(uint32_t _stageIx, uint32_t _ctxID, const uint8_t* _pcData, const core::SRange<const COpenGLSpecializedShader::SUniform>& _uniforms, const core::SRange<const GLint>& _locations) const
    {
        assert(_uniforms.size() == _locations.size());

        GLuint GLname = getShaderGLnameForCtx(_stageIx, _ctxID);
        uint8_t* state = getPushConstantsStateForStage(_stageIx, _ctxID);

        NBL_ASSUME_ALIGNED(_pcData, 128);

        using gl = COpenGLExtensionHandler;
        uint32_t loc_i = 0u;
        for(auto u_it = _uniforms.begin(); u_it != _uniforms.end(); ++u_it, ++loc_i)
        {
            if(_locations.begin()[loc_i] < 0)
                continue;

            const auto& u = *u_it;

            const auto& m = u.m;
            auto is_scalar_or_vec = [&m] { return (m.mtxRowCnt >= 1u && m.mtxColCnt == 1u); };
            auto is_mtx = [&m] { return (m.mtxRowCnt > 1u && m.mtxColCnt > 1u); };

            uint8_t* valueptr = state + m.offset;

            uint32_t arrayStride = m.arrayStride;
            // in case of non-array types, m.arrayStride is irrelevant
            // we should compute it though, so that we dont have to branch in the loop
            if(!m.isArray())
            {
                // 1N for scalar types, 2N for gvec2, 4N for gvec3 and gvec4
                // N==sizeof(float)
                // WARNING / TODO : need some touch in case when we want to support `double` push constants
                if(is_scalar_or_vec())
                    arrayStride = (m.mtxRowCnt == 1u) ? m.size : core::roundUpToPoT(m.mtxRowCnt) * sizeof(float);
                // same as size in case of matrices
                else if(is_mtx())
                    arrayStride = m.size;
            }
            assert(m.mtxStride == 0u || arrayStride % m.mtxStride == 0u);
            NBL_ASSUME_ALIGNED(valueptr, sizeof(float));
            //NBL_ASSUME_ALIGNED(valueptr, arrayStride); // should get the std140/std430 alignment of the type instead

            auto* baseOffset = _pcData + m.offset;
            NBL_ASSUME_ALIGNED(baseOffset, sizeof(float));
            //NBL_ASSUME_ALIGNED(baseOffset, arrayStride); // should get the std140/std430 alignment of the type instead

            constexpr uint32_t MAX_DWORD_SIZE = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE / sizeof(uint32_t);
            alignas(128u) std::array<uint32_t, MAX_DWORD_SIZE> packed_data;

            const uint32_t count = std::min<uint32_t>(m.count, MAX_DWORD_SIZE / (m.mtxRowCnt * m.mtxColCnt));
            if(!std::equal(baseOffset, baseOffset + arrayStride * count, valueptr) || !haveUniformsBeenEverSet(_stageIx, _ctxID))
            {
                // pack the constant data as OpenGL uniform update functions expect packed arrays
                {
                    const uint32_t rowOrColCnt = m.rowMajor ? m.mtxRowCnt : m.mtxColCnt;
                    const uint32_t len = m.rowMajor ? m.mtxColCnt : m.mtxRowCnt;
                    for(uint32_t i = 0u; i < count; ++i)
                        for(uint32_t c = 0u; c < rowOrColCnt; ++c)
                        {
                            auto in = reinterpret_cast<const uint32_t*>(baseOffset + i * arrayStride + c * m.mtxStride);
                            auto out = packed_data.data() + (i * m.mtxRowCnt * m.mtxColCnt) + (c * len);
                            std::copy(in, in + len, out);
                        }
                }

                if(is_mtx() && m.type == asset::EGVT_F32)
                {
                    PFNGLPROGRAMUNIFORMMATRIX4FVPROC glProgramUniformMatrixNxMfv_fptr[3][3]{
                        //N - num of columns, M - num of rows because of weird OpenGL naming convention
                        {&gl::extGlProgramUniformMatrix2fv, &gl::extGlProgramUniformMatrix2x3fv, &gl::extGlProgramUniformMatrix2x4fv},  //2xM
                        {&gl::extGlProgramUniformMatrix3x2fv, &gl::extGlProgramUniformMatrix3fv, &gl::extGlProgramUniformMatrix3x4fv},  //3xM
                        {&gl::extGlProgramUniformMatrix4x2fv, &gl::extGlProgramUniformMatrix4x3fv, &gl::extGlProgramUniformMatrix4fv}  //4xM
                    };

                    glProgramUniformMatrixNxMfv_fptr[m.mtxColCnt - 2u][m.mtxRowCnt - 2u](GLname, _locations.begin()[loc_i], count, m.rowMajor ? GL_TRUE : GL_FALSE, reinterpret_cast<const GLfloat*>(packed_data.data()));
                }
                else if(is_scalar_or_vec())
                {
                    switch(m.type)
                    {
                        case asset::EGVT_F32: {
                            PFNGLPROGRAMUNIFORM1FVPROC glProgramUniformNfv_fptr[4]{
                                &gl::extGlProgramUniform1fv, &gl::extGlProgramUniform2fv, &gl::extGlProgramUniform3fv, &gl::extGlProgramUniform4fv};
                            glProgramUniformNfv_fptr[m.mtxRowCnt - 1u](GLname, _locations.begin()[loc_i], count, reinterpret_cast<const GLfloat*>(packed_data.data()));
                            break;
                        }
                        case asset::EGVT_I32: {
                            PFNGLPROGRAMUNIFORM1IVPROC glProgramUniformNiv_fptr[4]{
                                &gl::extGlProgramUniform1iv, &gl::extGlProgramUniform2iv, &gl::extGlProgramUniform3iv, &gl::extGlProgramUniform4iv};
                            glProgramUniformNiv_fptr[m.mtxRowCnt - 1u](GLname, _locations.begin()[loc_i], count, reinterpret_cast<const GLint*>(packed_data.data()));
                            break;
                        }
                        case asset::EGVT_U32: {
                            PFNGLPROGRAMUNIFORM1UIVPROC glProgramUniformNuiv_fptr[4]{
                                &gl::extGlProgramUniform1uiv, &gl::extGlProgramUniform2uiv, &gl::extGlProgramUniform3uiv, &gl::extGlProgramUniform4uiv};
                            glProgramUniformNuiv_fptr[m.mtxRowCnt - 1u](GLname, _locations.begin()[loc_i], count, reinterpret_cast<const GLuint*>(packed_data.data()));
                            break;
                        }
                    }
                }
                std::copy(baseOffset, baseOffset + arrayStride * count, valueptr);
            }
        }
        afterUniformsSet(_stageIx, _ctxID);
    }

    GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t name_ix = _ctxID * _STAGE_COUNT + _stageIx;

        return (*m_GLprograms)[name_ix].GLname;
    }

    struct SShaderProgram
    {
        GLuint GLname = 0u;
        bool uniformsSetForTheVeryFirstTime = true;
    };
    //mutable for deferred GL objects creation
    mutable core::smart_refctd_dynamic_array<SShaderProgram> m_GLprograms;
    uint8_t* m_uniformValues;

private:
    bool haveUniformsBeenEverSet(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t ix = _ctxID * _STAGE_COUNT + _stageIx;
        return !(*m_GLprograms)[ix].uniformsSetForTheVeryFirstTime;
    }
    void afterUniformsSet(uint32_t _stageIx, uint32_t _ctxID) const
    {
        const uint32_t ix = _ctxID * _STAGE_COUNT + _stageIx;
        (*m_GLprograms)[ix].uniformsSetForTheVeryFirstTime = false;
    }
};

}
}
#endif

#endif
