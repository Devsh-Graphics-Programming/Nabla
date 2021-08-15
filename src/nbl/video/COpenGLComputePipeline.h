// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_OPENGL_COMPUTE_PIPELINE_H_INCLUDED__
#define __NBL_ASSET_C_OPENGL_COMPUTE_PIPELINE_H_INCLUDED__

#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IOpenGLPipeline.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl::video
{

class COpenGLComputePipeline : public IGPUComputePipeline, public IOpenGLPipeline<IGPUComputePipeline::SHADER_STAGE_COUNT>
{
    public:
        COpenGLComputePipeline(
            core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& device, IOpenGL_FunctionTable* _gl,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<IGPUSpecializedShader>&& _cs,
            uint32_t _ctxCount, uint32_t _ctxID, GLuint _GLname, const COpenGLSpecializedShader::SProgramBinary& _binary
        ) : IGPUComputePipeline(core::smart_refctd_ptr(device), std::move(_layout), std::move(_cs)), 
            IOpenGLPipeline(device.get(), _gl, _ctxCount, _ctxID, &_GLname, &_binary),
            m_lastUpdateStamp(0u)
        {

        }

        bool containsShader() const { return static_cast<bool>(m_shader); }

        GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
        {
            return IOpenGLPipeline<1>::getShaderGLnameForCtx(_stageIx, _ctxID);
        }
        
        struct alignas(128) PushConstantsState
        {
	        alignas(128) uint8_t data[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
	        core::smart_refctd_ptr<const COpenGLPipelineLayout> layout;
	        std::atomic_uint32_t stageUpdateStamps[IGPUComputePipeline::SHADER_STAGE_COUNT] = { 0u };

	        inline uint32_t getStamp(IGPUSpecializedShader::E_SHADER_STAGE _stage) const
	        {
		        assert(_stage == IGPUSpecializedShader::ESS_COMPUTE);
		        return stageUpdateStamps[0u];
	        }
	        inline void incrementStamps(uint32_t _stages)
	        {
		        if (_stages & IGPUSpecializedShader::ESS_COMPUTE)
                    stageUpdateStamps[0u]++;
	        }
        };
        inline void setUniformsImitatingPushConstants(IOpenGL_FunctionTable* gl, uint32_t _ctxID, const PushConstantsState& _pcState) const
        {
            uint32_t stampValue = _pcState.getStamp(IGPUSpecializedShader::ESS_COMPUTE);
            if (stampValue>m_lastUpdateStamp)
            {
                auto uniforms = static_cast<COpenGLSpecializedShader*>(m_shader.get())->getUniforms();
                auto locations = static_cast<COpenGLSpecializedShader*>(m_shader.get())->getLocations();
                if (uniforms.size())
                    IOpenGLPipeline<1>::setUniformsImitatingPushConstants(gl, 0u, _ctxID, _pcState.data, uniforms, locations);
                m_lastUpdateStamp = stampValue;
            }
        }

    protected:
        virtual ~COpenGLComputePipeline();

        mutable uint32_t m_lastUpdateStamp;
};

}

#endif

#endif