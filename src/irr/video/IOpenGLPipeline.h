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

    uint8_t* getPushConstantsStateForStage(uint32_t _stageIx, uint32_t _ctxID) const { return const_cast<uint8_t*>(m_uniformValues + ((_STAGE_COUNT*_ctxID + _stageIx)*IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)); }

protected:
    void setUniformsImitatingPushConstants(uint32_t _stageIx, uint32_t _ctxID, const uint8_t* _pcData, const core::SRange<const COpenGLSpecializedShader::SUniform>& _uniforms, const core::SRange<const GLint>& _locations) const
    {
        assert(_uniforms.length()==_locations.length());

        GLuint GLname = getShaderGLnameForCtx(_stageIx, _ctxID);
        uint8_t* state = getPushConstantsStateForStage(_stageIx, _ctxID);

        IRR_ASSUME_ALIGNED(_pcData, 128);

        using gl = COpenGLExtensionHandler;
	    uint32_t loc_i = 0u;
        for (auto u_it=_uniforms.begin(); u_it!=_uniforms.end(); ++u_it, ++loc_i)
        {
		    if (_locations.begin()[loc_i]<0)
			    continue;

		    const auto& u = *u_it;

            const auto& m = u.m;
            auto is_scalar_or_vec = [&m] { return (m.mtxRowCnt >= 1u && m.mtxColCnt == 1u); };
            auto is_mtx = [&m] { return (m.mtxRowCnt > 1u && m.mtxColCnt > 1u); };

		    uint8_t* valueptr = state+m.offset;

            uint32_t arrayStride = 0u;
            {
                uint32_t arrayStride1 = 0u;
                if (is_scalar_or_vec())
                    arrayStride1 = (m.mtxRowCnt==1u) ? m.size : core::roundUpToPoT(m.mtxRowCnt)*4u;
                else if (is_mtx())
                    arrayStride1 = m.arrayStride;
                assert(arrayStride1);
                arrayStride = (m.count <= 1u) ? arrayStride1 : m.arrayStride;
            }
		    assert(m.mtxStride==0u || arrayStride%m.mtxStride==0u);
		    IRR_ASSUME_ALIGNED(valueptr, sizeof(float));
		    IRR_ASSUME_ALIGNED(valueptr, arrayStride);
		
		    auto* baseOffset = _pcData+m.offset;
		    IRR_ASSUME_ALIGNED(baseOffset, sizeof(float));
		    IRR_ASSUME_ALIGNED(baseOffset, arrayStride);

		    constexpr uint32_t MAX_DWORD_SIZE = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE/sizeof(GLfloat);
		    alignas(128u) std::array<GLfloat,MAX_DWORD_SIZE> packed_data;

            const uint32_t count = std::min<uint32_t>(m.count, MAX_DWORD_SIZE/(m.count*m.mtxRowCnt*m.mtxColCnt));
		    if (!std::equal(baseOffset, baseOffset+arrayStride*count, valueptr) || !haveUniformsBeenEverSet(_stageIx, _ctxID))
		    {
			    // pack the constant data as OpenGL uniform update functions expect packed arrays
			    {
				    const bool isRowMajor = is_scalar_or_vec() || m.rowMajor;
				    const uint32_t rowOrColCnt = isRowMajor ? m.mtxColCnt : m.mtxRowCnt;
				    const uint32_t len = isRowMajor ? m.mtxRowCnt : m.mtxColCnt;
				    for (uint32_t i = 0u; i < count; ++i)
				    for (uint32_t c = 0u; c < rowOrColCnt; ++c)
				    {
					    const GLfloat* in = reinterpret_cast<const GLfloat*>(baseOffset + i*arrayStride + c*m.mtxStride);
					    GLfloat* out = packed_data.data() + (i*m.mtxRowCnt*m.mtxColCnt) + (c*len);
					    std::copy(in, in+len, out);
				    }
			    }

			    if (is_mtx() && m.type==asset::EGVT_F32)
			    {
					    PFNGLPROGRAMUNIFORMMATRIX4FVPROC glProgramUniformMatrixNxMfv_fptr[3][3]{ //N - num of columns, M - num of rows because of weird OpenGL naming convention
						    {&gl::extGlProgramUniformMatrix2fv, &gl::extGlProgramUniformMatrix2x3fv, &gl::extGlProgramUniformMatrix2x4fv},//2xM
						    {&gl::extGlProgramUniformMatrix3x2fv, &gl::extGlProgramUniformMatrix3fv, &gl::extGlProgramUniformMatrix3x4fv},//3xM
						    {&gl::extGlProgramUniformMatrix4x2fv, &gl::extGlProgramUniformMatrix4x3fv, &gl::extGlProgramUniformMatrix4fv} //4xM
					    };

					    glProgramUniformMatrixNxMfv_fptr[m.mtxColCnt-2u][m.mtxRowCnt-2u](GLname, _locations.begin()[loc_i], m.count, m.rowMajor ? GL_TRUE : GL_FALSE, packed_data.data());
			    }
			    else if (is_scalar_or_vec())
			    {
				    switch (m.type) 
				    {
					    case asset::EGVT_F32:
					    {
						    PFNGLPROGRAMUNIFORM1FVPROC glProgramUniformNfv_fptr[4]{
							    &gl::extGlProgramUniform1fv, &gl::extGlProgramUniform2fv, &gl::extGlProgramUniform3fv, &gl::extGlProgramUniform4fv
						    };
						    glProgramUniformNfv_fptr[m.mtxRowCnt-1u](GLname, _locations.begin()[loc_i], m.count, packed_data.data());
						    break;
					    }
					    case asset::EGVT_I32:
					    {
						    PFNGLPROGRAMUNIFORM1IVPROC glProgramUniformNiv_fptr[4]{
							    &gl::extGlProgramUniform1iv, &gl::extGlProgramUniform2iv, &gl::extGlProgramUniform3iv, &gl::extGlProgramUniform4iv
						    };
						    glProgramUniformNiv_fptr[m.mtxRowCnt-1u](GLname, _locations.begin()[loc_i], m.count, reinterpret_cast<const GLint*>(packed_data.data()));
						    break;
					    }
					    case asset::EGVT_U32:
					    {
						    PFNGLPROGRAMUNIFORM1UIVPROC glProgramUniformNuiv_fptr[4]{
							    &gl::extGlProgramUniform1uiv, &gl::extGlProgramUniform2uiv, &gl::extGlProgramUniform3uiv, &gl::extGlProgramUniform4uiv
						    };
						    glProgramUniformNuiv_fptr[m.mtxRowCnt-1u](GLname, _locations.begin()[loc_i], m.count, reinterpret_cast<const GLuint*>(packed_data.data()));
						    break;
					    }
				    }
			    }
			    std::copy(baseOffset, baseOffset+arrayStride*count, valueptr);
            }
        }
        afterUniformsSet(_stageIx, _ctxID);
    }

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

}}

#endif