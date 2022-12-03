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

struct uvec4
{
    uint32_t x, y, z, w;
};
#include "nbl/builtin/glsl/ext/CentralLimitBoxBlur/parameters_struct.glsl"

class CBlurPerformer final : public core::IReferenceCounted
{
public:
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t DefaultWorkgroupSize = 256u;
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t PassesPerAxis = 3u;

    typedef nbl_glsl_ext_Blur_Parameters_t Parameters_t;

    struct DispatchInfo_t
    {
        uint32_t wg_count[3];
    };

    CBlurPerformer(video::ILogicalDevice* device, uint32_t maxDimensionSize, bool useHalfStorage);

    static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

    static core::smart_refctd_ptr<video::IGPUSpecializedShader> createSpecializedShader(const char* shaderIncludePath, const uint32_t axisDim, const bool useHalfStorage, video::ILogicalDevice* device);

    //! Returns the required size of a buffer needed to hold the output of one pass.
    //! Typically used as an intermediate storage.
    //! For example, for a 2D blur, use a buffer of the reported size to hold the result of the 1st pass/dispatch --input to the second pass/dispatch.
    static inline size_t getPassOutputBufferSize(const asset::VkExtent3D& dims, const uint32_t channelCount)
    {
        return dims.width * dims.height * dims.depth * channelCount * sizeof(float);
    }

    static inline uint32_t buildParameters(uint32_t numChannels, const asset::VkExtent3D& inputDimensions, Parameters_t* outParams, DispatchInfo_t* outInfos,
        const float radius, const asset::ISampler::E_TEXTURE_CLAMP* wrappingType, const asset::ISampler::E_TEXTURE_BORDER_COLOR* borderColor = nullptr)
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
                params.input_dimensions.w = (passAxis << 30) | (numChannels << 27) | (wrappingType[i] << 25);

                if (borderColor)
                {
                    assert(borderColor[i] <= asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_INT_OPAQUE_WHITE);
                    params.input_dimensions.w |= borderColor[i] << 22;
                }
        
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

    static inline void dispatchHelper(video::IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipelineLayout, const Parameters_t& pushConstants, const DispatchInfo_t& dispatchInfo)
    {
        cmdbuf->pushConstants(pipelineLayout, asset::IShader::ESS_COMPUTE, 0, sizeof(Parameters_t), &pushConstants);
        cmdbuf->dispatch(dispatchInfo.wg_count[0], dispatchInfo.wg_count[1], dispatchInfo.wg_count[2]);
    }

    inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultDescriptorSetLayout() const { return m_dsLayout; }
    inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout() const { return m_pplnLayout; }
    inline core::smart_refctd_ptr<video::IGPUComputePipeline> getDefaultPipeline() const { return m_ppln; }

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
