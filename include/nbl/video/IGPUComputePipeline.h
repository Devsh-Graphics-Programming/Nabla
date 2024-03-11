// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/IPipeline.h"

#include "nbl/video/SPipelineCreationParams.h"
#include "nbl/video/SPipelineCreationParams.h"


namespace nbl::video
{

class IGPUComputePipeline : public IBackendObject, public asset::IPipeline<const IGPUPipelineLayout>
{
        using pipeline_t = asset::IPipeline<const IGPUPipelineLayout>;

    public:
        struct SCreationParams final : pipeline_t::SCreationParams, SPipelineCreationParams<const IGPUComputePipeline>
        {
            // By construction we satisfy from:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-03365
            // to:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-04945
            // and:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07367
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07996
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::SCreationParams::FLAGS::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                DISPATCH_BASE = 1<<4,
                CAPTURE_STATISTICS = base_flag(CAPTURE_STATISTICS),
                CAPTURE_INTERNAL_REPRESENTATIONS = base_flag(CAPTURE_INTERNAL_REPRESENTATIONS),
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
                LINK_TIME_OPTIMIZATION = base_flag(LINK_TIME_OPTIMIZATION),
                RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT = base_flag(RETAIN_LINK_TIME_OPTIMIZATION_INFO),
                // Not Supported Yet
                //CREATE_LIBRARY = base_flag(CREATE_LIBRARY),
                // Not Supported Yet
                //INDIRECT_BINDABLE_NV = base_flag(INDIRECT_BINDABLE_NV),
            };
            #undef base_flag

            inline SSpecializationValidationResult valid() const
            {
                const int32_t dataSize = shader.valid();
                if (dataSize<0)
                    return {};
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-stage-00701
                if (!layout || shader.shader->getStage()!=IGPUShader::ESS_COMPUTE)
                    return {};

                uint32_t count = 0;
                if (shader.entries)
                {
                    if (shader.entries->size()>0x7fffffff)
                        return {};
                    count = static_cast<uint32_t>(shader.entries->size());
                }
                return {.count=dataSize ? count:0,.dataSize=static_cast<uint32_t>(dataSize)};
            }

            inline std::span<const IGPUShader::SSpecInfo> getShaders() const {return {&shader,1}; }

            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
            IGPUShader::SSpecInfo shader = {};
        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const {return m_flags;}

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

    protected:
        inline IGPUComputePipeline(core::smart_refctd_ptr<const IGPUPipelineLayout>&& _layout, const core::bitflag<SCreationParams::FLAGS> _flags) :
            IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>(_layout->getOriginDevice())), pipeline_t(std::move(_layout)), m_flags(_flags) {}
        virtual ~IGPUComputePipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};
NBL_ENUM_ADD_BITWISE_OPERATORS(IGPUComputePipeline::SCreationParams::FLAGS)

}

#endif