// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IComputePipeline.h"

#include "nbl/video/IGPUPipeline.h"
#include "nbl/video/SPipelineCreationParams.h"


namespace nbl::video
{

class IGPUComputePipeline : public IGPUPipeline<asset::IComputePipeline<const IGPUPipelineLayout>>
{
        using pipeline_t = asset::IComputePipeline<const IGPUPipelineLayout>;

    public:
        struct SCreationParams final : SPipelineCreationParams<const IGPUComputePipeline>
        {
            // By construction we satisfy from:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-03365
            // to:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-04945
            // and:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07367
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07996
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::FLAGS::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                DISPATCH_BASE = 1<<4,
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
                // Not Supported Yet
                //CREATE_LIBRARY = base_flag(CREATE_LIBRARY),
                // Not Supported Yet
                //INDIRECT_BINDABLE_NV = base_flag(INDIRECT_BINDABLE_NV),
            };
            #undef base_flag

            inline SSpecializationValidationResult valid() const
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-stage-00701
                if (!layout)
                    return {};

                SSpecializationValidationResult retval = {
                    .count = 0,
                    .dataSize = 0,
                };

                if (!shader.shader)
                    return {};

                if (!shader.accumulateSpecializationValidationResult(&retval))
                    return {};

                return retval;
            }

            inline core::bitflag<hlsl::ShaderStage> getRequiredSubgroupStages() const
            {
                if (shader.shader && shader.requiredSubgroupSize >= asset::IPipelineBase::SUBGROUP_SIZE::REQUIRE_4)
                {
                    return hlsl::ESS_COMPUTE;
                }
                return {};
            }

            inline core::bitflag<FLAGS>& getFlags() { return flags; }

            inline core::bitflag<FLAGS> getFlags() const { return flags; }

            const IGPUPipelineLayout* layout = nullptr;
            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
            SCachedCreationParams cached = {};
            SShaderSpecInfo shader = {};
        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const {return m_flags;}

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

    protected:
        inline IGPUComputePipeline(const SCreationParams& params) :
          IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice()), params.layout, params.cached), m_flags(params.flags)
        {}
        virtual ~IGPUComputePipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};
NBL_ENUM_ADD_BITWISE_OPERATORS(IGPUComputePipeline::SCreationParams::FLAGS)

}

#endif