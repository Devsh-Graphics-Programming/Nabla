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
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::CreationFlags::F)
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

            template<typename ExtraLambda>
            inline bool impl_valid(ExtraLambda&& extra) const
            {
                if (!layout)
                    return false;

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576
                if (!renderpass || cached.subpassIx>=renderpass->getSubpassCount())
                    return false;

                // TODO: check rasterization samples, etc.
                //rp->getCreationParameters().subpasses[i]

                core::bitflag<hlsl::ShaderStage> stagePresence = {};
                for (auto shader_i = 0u; shader_i < shaders.size(); shader_i++)
                {
                    const auto& info = shaders[shader_i];
                    if (!extra(info))
                        return false;
                    if (info.shader)
                        stagePresence |= indexToStage(shader_i);
                }

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-stage-02096
                if (!stagePresence.hasFlags(hlsl::ShaderStage::ESS_VERTEX))
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pStages-00729
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pStages-00730
                if (stagePresence.hasFlags(hlsl::ShaderStage::ESS_TESSELLATION_CONTROL)!=stagePresence.hasFlags(hlsl::ShaderStage::ESS_TESSELLATION_EVALUATION))
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pStages-08888
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-topology-08889
                if (stagePresence.hasFlags(hlsl::ShaderStage::ESS_TESSELLATION_EVALUATION)!=(cached.primitiveAssembly.primitiveType==asset::EPT_PATCH_LIST))
                    return false;
                
                return true;
            }

            inline SSpecializationValidationResult valid() const
            {
                if (!layout)
                    return {};
                SSpecializationValidationResult retval = {.count=0,.dataSize=0};
                const bool valid = impl_valid([&retval](const SShaderSpecInfo& info)->bool
                {
                    const auto dataSize = info.valid();
                    if (dataSize<0)
                        return false;
                    else if (dataSize==0)
                        return true;

                    const size_t count = info.entries ? info.entries->size():0x80000000ull;
                    if (count>0x7fffffff)
                        return {};
                    retval += {.count=dataSize ? static_cast<uint32_t>(count):0,.dataSize=static_cast<uint32_t>(dataSize)};
                    return retval;
                });
                if (!valid)
                    return {};
                return retval;
            }

            inline std::span<const SShaderSpecInfo> getShaders() const {return shaders;}

            IGPUPipelineLayout* layout = nullptr;
            std::span<const SShaderSpecInfo> shaders = {};
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
        virtual ~IGPUGraphicsPipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};

}

#endif