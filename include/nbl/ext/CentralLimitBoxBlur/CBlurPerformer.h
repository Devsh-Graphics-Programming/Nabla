// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_
#define _NBL_EXT_BLUR_C_BLUR_PERFORMER_INCLUDED_

#include "nabla.h"

namespace nbl
{
namespace ext
{
namespace CentralLimitBoxBlur
{

typedef uint32_t uint;
struct alignas(16) uvec4
{
    uint x, y, z, w;
};
#include "nbl/builtin/glsl/ext/CentralLimitBoxBlur/parameters_struct.glsl"

class CBlurPerformer final : public core::IReferenceCounted
{
public:
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORKGROUP_SIZE = 256u;
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t PASSES_PER_AXIS = 3u;

    typedef nbl_glsl_ext_Blur_Parameters_t Parameters_t;

    struct DispatchInfo_t
    {
        uint32_t wg_count[3];
    };

    CBlurPerformer(video::IVideoDriver* driver, uint32_t maxDimensionSize, bool useHalfStorage);

    static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

    static inline void defaultBarrier()
    {
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    static inline size_t getOutputBufferSize(const asset::VkExtent3D& inputDimensions, const uint32_t channelCount)
    {
        return inputDimensions.width * inputDimensions.height * inputDimensions.depth * channelCount * sizeof(float);
    }

    static inline uint32_t buildParameters(uint32_t numChannels, const asset::VkExtent3D& inputDimensions, Parameters_t* outParams, DispatchInfo_t* outInfos,
        const asset::ISampler::E_TEXTURE_CLAMP* wrappingType, const float radius)
    {
        uint32_t passesRequired = 0u;
        
        if (numChannels)
        {
            for (uint32_t i = 0u; i < 3u; i++)
            {
                auto dim = (&inputDimensions.width)[i];
                if (dim >= 2u)
                    passesRequired++;
            }
        }

        if (passesRequired)
        {
            for (uint32_t i = 0u; i < passesRequired; i++)
            {
                auto& params = outParams[i];

                params.input_dimensions.x = inputDimensions.width;
                params.input_dimensions.y = inputDimensions.height;
                params.input_dimensions.z = inputDimensions.depth;
        
                assert(wrappingType[i] <= asset::ISampler::ETC_MIRROR);

                const auto passAxis = i;
                const auto axisLen = (&inputDimensions.width)[passAxis]; // Todo(achal): For virtual threads calculation you need to send log2 of axis dim
                params.input_dimensions.w = (passAxis << 30) | (numChannels << 28) | (wrappingType[i] << 26);
        
                params.input_strides.x = 1u;
                params.input_strides.y = inputDimensions.width;
                params.input_strides.z = params.input_strides.y * inputDimensions.height;
                params.input_strides.w = params.input_strides.z * inputDimensions.depth;

                // At this point I'm wondering do you even need both input and output strides
                params.output_strides = params.input_strides;

                params.radius = radius;

                auto& dispatch = outInfos[i];
                dispatch.wg_count[0] = inputDimensions.width;
                dispatch.wg_count[1] = inputDimensions.height;
                dispatch.wg_count[2] = inputDimensions.depth;
                dispatch.wg_count[passAxis] = 1u;
            }
        }
        
        return passesRequired;
    }

    static inline void dispatchHelper(video::IVideoDriver* driver, const video::IGPUPipelineLayout* pipelineLayout, const Parameters_t& params,
        const DispatchInfo_t& dispatchInfo, bool issueDefaultBarrier = true)
    {
        driver->pushConstants(pipelineLayout, video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(Parameters_t), &params);
        driver->dispatch(dispatchInfo.wg_count[0], dispatchInfo.wg_count[1], dispatchInfo.wg_count[2]);

        if (issueDefaultBarrier)
            defaultBarrier();
    }

private:
    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_dsLayout = nullptr;
    core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pplnLayout = nullptr;
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_ppln = nullptr;

    uint32_t m_maxBlurLen;
    bool m_halfFloatStorage;
};

}
}
}

#endif
