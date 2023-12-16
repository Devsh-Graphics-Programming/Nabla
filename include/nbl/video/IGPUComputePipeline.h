// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/IComputePipeline.h"

#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IPipeline.h"


namespace nbl::video
{

class IGPUComputePipeline : public IPipeline<IGPUComputePipeline>, public asset::IComputePipeline<const IGPUSpecializedShader>
{
        using pipeline_t = IPipeline<IGPUComputePipeline>;
        using base_t = asset::IComputePipeline<const IGPUSpecializedShader>;

    public:
        struct SCreationParams final : pipeline_t::SCreationParams
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

            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
            const IGPUSpecializedShader* shader = nullptr;
        };

        inline SCreationParams::FLAGS getCreationFlags() const {return m_flags;}

    protected:
        inline IGPUComputePipeline(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUSpecializedShader>&& _cs, const SCreationParams::FLAGS _flags) :
            pipeline_t(std::move(dev)), base_t(std::move(_cs)), m_flags(_flags) {}
        virtual ~IGPUComputePipeline() = default;

        const SCreationParams::FLAGS m_flags;
};
NBL_ENUM_ADD_BITWISE_OPERATORS(IGPUComputePipeline::SCreationParams::FLAGS)

}

#endif