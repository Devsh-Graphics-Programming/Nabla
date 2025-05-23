#ifndef _NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/asset/IGraphicsPipeline.h"

#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUPipeline.h"


namespace nbl::video
{

class IGPUGraphicsPipeline : public IGPUPipeline<asset::IGraphicsPipeline<const IGPUPipelineLayout, const IGPURenderpass>>
{
        using pipeline_t = asset::IGraphicsPipeline<const IGPUPipelineLayout,const IGPURenderpass>;

    public:
        struct SCreationParams final : public SPipelineCreationParams<const IGPUGraphicsPipeline>
        {
            public:
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::FLAGS::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
            };
            #undef base_flag

            inline SSpecializationValidationResult valid() const
            {
                if (!layout)
                    return {};
                SSpecializationValidationResult retval = {.count=0,.dataSize=0};
                if (!layout)
                    return {};

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576
                if (!renderpass || cached.subpassIx>=renderpass->getSubpassCount())
                    return {};

                // TODO: check rasterization samples, etc.
                //rp->getCreationParameters().subpasses[i]

                core::bitflag<hlsl::ShaderStage> stagePresence = {};

                auto processSpecInfo = [&](const SShaderSpecInfo& specInfo, hlsl::ShaderStage stage)
                {
                    if (!specInfo.shader) return true;
                    if (!specInfo.accumulateSpecializationValidationResult(&retval)) return false;
                    stagePresence |= stage;
                    return true;
                };
                if (!processSpecInfo(vertexShader, hlsl::ShaderStage::ESS_VERTEX)) return {};
                if (!processSpecInfo(tesselationControlShader, hlsl::ShaderStage::ESS_TESSELLATION_CONTROL)) return {};
                if (!processSpecInfo(tesselationEvaluationShader, hlsl::ShaderStage::ESS_TESSELLATION_EVALUATION)) return {};
                if (!processSpecInfo(geometryShader, hlsl::ShaderStage::ESS_GEOMETRY)) return {};
                if (!processSpecInfo(fragmentShader, hlsl::ShaderStage::ESS_FRAGMENT)) return {};
                
                if (!hasRequiredStages(stagePresence, cached.primitiveAssembly.primitiveType))
                    return {};
                return retval;
            }

            IGPUPipelineLayout* layout = nullptr;
            SShaderSpecInfo vertexShader;
            SShaderSpecInfo tesselationControlShader;
            SShaderSpecInfo tesselationEvaluationShader;
            SShaderSpecInfo geometryShader;
            SShaderSpecInfo fragmentShader;
            SCachedCreationParams cached = {};
            renderpass_t* renderpass = nullptr;

            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const {return m_flags;}

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

    protected:
        IGPUGraphicsPipeline(const SCreationParams& params) :
          IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice()), params.layout, params.cached, params.renderpass), m_flags(params.flags)
        {}
        virtual ~IGPUGraphicsPipeline() override = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};

}

#endif