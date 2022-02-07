// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_OPENGL_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __NBL_ASSET_C_OPENGL_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "nbl/video/IGPURenderpassIndependentPipeline.h"

#include "COpenGLExtensionHandler.h"
#include "COpenGLSpecializedShader.h"

#include "nbl/video/IOpenGLPipeline.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
{
namespace video
{
class COpenGLRenderpassIndependentPipeline final : public IGPURenderpassIndependentPipeline, public IOpenGLPipeline<IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT>
{
public:
    //! _binaries' elements are getting move()'d!
    COpenGLRenderpassIndependentPipeline(
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        IGPUSpecializedShader* const* _shadersBegin, IGPUSpecializedShader* const* _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams,
        const asset::SBlendParams& _blendParams,
        const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams,
        uint32_t _ctxCount, uint32_t _ctxID, const GLuint _GLnames[SHADER_STAGE_COUNT], const COpenGLSpecializedShader::SProgramBinary _binaries[SHADER_STAGE_COUNT])
        : IGPURenderpassIndependentPipeline(
              std::move(_layout), _shadersBegin, _shadersEnd,
              _vertexInputParams, _blendParams, _primAsmParams, _rasterParams),
          IOpenGLPipeline(_ctxCount, _ctxID, _GLnames, _binaries),
          m_stagePresenceMask(0u)
    {
        static_assert(asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT == asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT, "This code below has to be divided into 2 loops");
        static_assert(asset::EF_UNKNOWN <= 0xffu, "All E_FORMAT values must fit in 1 byte or hash falls apart");
        static_assert(asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT == 16u && asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT == 16u, "m_vaoHashval.mapAttrToBinding needs adjustments");
        for(size_t i = 0ull; i < asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
        {
            if(!((m_vertexInputParams.enabledAttribFlags >> i) & 1u))
            {
                m_vaoHashval.attribFormatAndComponentCount[i] = asset::EF_UNKNOWN;
                continue;
            }

            m_vaoHashval.attribFormatAndComponentCount[i] = m_vertexInputParams.attributes[i].format;
            m_vaoHashval.setRelativeOffsetForAttrib(i, m_vertexInputParams.attributes[i].relativeOffset);

            const uint64_t bnd = m_vertexInputParams.attributes[i].binding;
            m_vaoHashval.mapAttrToBinding |= (bnd << (i * 4));
            m_vaoHashval.setStrideForBinding(bnd, m_vertexInputParams.bindings[bnd].stride);
            m_vaoHashval.divisors |= ((m_vertexInputParams.bindings[bnd].inputRate == asset::EVIR_PER_VERTEX ? 0u : 1u) << bnd);
        }
        for(uint32_t i = 0u; i < SHADER_STAGE_COUNT; ++i)
        {
            const bool present = static_cast<bool>(m_shaders[i]);
            m_stagePresenceMask |= (static_cast<uint32_t>(present) << i);
            m_lastUpdateStamp[i] = 0u;
        }
    }

    // should be called in case of absence of GL_ARB_shader_draw_parameters only
    void setBaseInstanceUniform(uint32_t _ctxID, GLint _baseInstance) const
    {
        // only this function touches this uniform
        constexpr const char* SPIRV_CROSS_BaseInstanceUniformName = "SPIRV_Cross_BaseInstance";

        GLint& value = getBaseInstanceState(_ctxID)->cache;
        if(value == _baseInstance)
            return;

        const GLuint programID = getShaderGLnameForCtx(ESSI_VERTEX_SHADER_IX, _ctxID);
        GLint& uid = getBaseInstanceState(_ctxID)->id;
        if(uid == -1)
        {
            uid = COpenGLExtensionHandler::extGlGetUniformLocation(programID, SPIRV_CROSS_BaseInstanceUniformName);
        }
        if(uid == -1)
            return;

        value = _baseInstance;

        COpenGLExtensionHandler::extGlProgramUniform1iv(programID, uid, 1u, &value);
    }

    uint32_t getStagePresenceMask() const { return m_stagePresenceMask; }

    GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
    {
        if(!m_shaders[_stageIx])
            return 0u;

        return IOpenGLPipeline<SHADER_STAGE_COUNT>::getShaderGLnameForCtx(_stageIx, _ctxID);
    }

    struct PushConstantsState
    {
        alignas(128) uint8_t data[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
        core::smart_refctd_ptr<const COpenGLPipelineLayout> layout;
        std::atomic_uint32_t stageUpdateStamps[IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT] = {0u};

        inline uint32_t getStamp(IGPUSpecializedShader::E_SHADER_STAGE _stage) const
        {
            const uint32_t ix = core::findLSB<std::underlying_type_t<IGPUSpecializedShader::E_SHADER_STAGE>>(_stage);
            assert(ix < IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT);
            return stageUpdateStamps[ix];
        }
        inline void incrementStamps(uint32_t _stages)
        {
            for(uint32_t i = 0u; i < IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++i)
                if((_stages >> i) & 1u)
                    ++stageUpdateStamps[i];
        }
    };
    inline void setUniformsImitatingPushConstants(uint32_t _ctxID, const PushConstantsState& _pcState) const
    {
        for(uint32_t i = 0u; i < SHADER_STAGE_COUNT; ++i)
        {
            auto stage = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(1u << i);
            if((m_stagePresenceMask & stage) == 0u)
                continue;

            uint32_t stampValue = _pcState.getStamp(stage);
            if(stampValue > m_lastUpdateStamp[i])
            {
                auto uniforms = static_cast<COpenGLSpecializedShader*>(m_shaders[i].get())->getUniforms();
                auto locations = static_cast<COpenGLSpecializedShader*>(m_shaders[i].get())->getLocations();
                if(uniforms.size())
                    IOpenGLPipeline<SHADER_STAGE_COUNT>::setUniformsImitatingPushConstants(i, _ctxID, _pcState.data, uniforms, locations);
                m_lastUpdateStamp[i] = stampValue;
            }
        }
    }

    struct SVAOHash
    {
        constexpr static size_t getHashLength()
        {
            return sizeof(hashVal) / sizeof(uint32_t);
        }

        inline bool operator!=(const SVAOHash& rhs) const
        {
            for(size_t i = 0u; i < getHashLength(); ++i)
                if(hashVal[i] != rhs.hashVal[i])
                    return true;
            return false;
        }
        inline bool operator==(const SVAOHash& rhs) const
        {
            return !((*this) != rhs);
        }
        inline bool operator<(const SVAOHash& rhs) const
        {
            for(size_t i = 0u; i < getHashLength(); ++i)
                if(hashVal[i] < rhs.hashVal[i])
                    return true;
            return false;
        }

        inline uint32_t getRelativeOffsetForAttrib(uint32_t _attr) const
        {
            return extract12bits(_attr, relOffsets);
        }
        inline void setRelativeOffsetForAttrib(uint32_t _attr, uint64_t _val)
        {
            return set12bits(_attr, relOffsets, _val);
        }
        inline uint32_t getStrideForBinding(uint32_t _bnd) const
        {
            return extract12bits(_bnd, strides);
        }
        inline void setStrideForBinding(uint32_t _bnd, uint64_t _val)
        {
            return set12bits(_bnd, strides, _val);
        }

        inline uint32_t getBindingForAttrib(uint32_t _attr) const
        {
            const uint32_t shift = 4u * _attr;
            return (mapAttrToBinding >> shift) & 0xfull;
        }

        inline uint32_t getDivisorForBinding(uint32_t _bnd) const
        {
            return (divisors >> _bnd) & 1u;
        }

        union
        {
#include "nbl/nblpack.h"
            struct
            {
                uint64_t relOffsets[3];  //16*12 bits
                uint64_t strides[3];  //16*12 bits
                uint64_t mapAttrToBinding;  //16*4 bits
                uint16_t divisors;
                //E_FORMAT values
                uint8_t attribFormatAndComponentCount[asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT];  //attribute X is enabled if attribFormatAndComponentCount[X]!=EF_UNKNOWN
            } PACK_STRUCT;
#include "nbl/nblunpack.h"
            uint32_t hashVal[19]{};
        };

    private:
        inline static uint32_t extract12bits(uint32_t _ix, const uint64_t _bits[3])
        {
            if(_ix == 5u)
            {
                uint32_t val = (_bits[0] & (0xfull << 60)) >> 60;
                val |= (_bits[1] & 0xffull) << 4;
                return val;
            }
            if(_ix == 11u)
            {
                uint32_t val = (_bits[1] & (0xffull << 56)) >> 56;
                val |= (_bits[2] & 0xfull) << 8;
                return val;
            }
            const uint32_t ix = (_ix > 5u) + (_ix > 11u);
            const uint32_t shift = (_ix * 12u) - (ix * 64u);

            return (_bits[ix] >> shift) & 0xfffull;
        }
        inline static void set12bits(uint32_t _ix, uint64_t _bits[3], uint64_t _val)
        {
            assert(!(_val & (~0xfffull)));  //bits higher than [0..11] must not be set
            if(_ix == 5u)
            {
                _bits[0] &= (~(0xfull << 60));
                _bits[0] |= (_val << 60);
                _bits[1] &= (~0xffull);
                _bits[1] |= (_val >> 4);
                return;
            }
            if(_ix == 11u)
            {
                _bits[1] &= (~(0xffull << 56));
                _bits[1] |= (_val << 56);
                _bits[2] &= (~0xfull);
                _bits[2] |= (_val >> 8);
                return;
            }

            const uint32_t ix = (_ix > 5u) + (_ix > 11u);
            const uint32_t shift = (_ix * 12u) - (ix * 64u);
            _bits[ix] &= (~(0xfffull << shift));
            _bits[ix] |= (_val << shift);
        }
    };

    const SVAOHash& getVAOHash() const { return m_vaoHashval; }

protected:
    ~COpenGLRenderpassIndependentPipeline() = default;

private:
    SVAOHash m_vaoHashval;
    uint32_t m_stagePresenceMask;
    mutable uint32_t m_lastUpdateStamp[SHADER_STAGE_COUNT];
};

}
}
#endif

#endif
