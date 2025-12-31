#ifndef _NBL_I_GPU_MESH_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_MESH_PIPELINE_H_INCLUDED_

#include "nbl/asset/IMeshPipeline.h"

#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUPipeline.h"

//related spec

//i feel like this MIGHT get stuffed into graphicspipeline but idk

/*
https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-PrimitiveId-06264
** If the pipeline requires pre-rasterization shader state, it includes a mesh shader and the fragment shader code reads from an input variable that is decorated with PrimitiveId, then the mesh shader code must write to a matching output variable, decorated with PrimitiveId, in all execution paths

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-07064
* If renderPass is not VK_NULL_HANDLE, the pipeline is being created with pre-rasterization shader state, subpass viewMask is not 0, and multiviewMeshShader is not enabled, then pStages must not include a mesh shader

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-None-02322
* If the pipeline requires pre-rasterization shader state, and there are any mesh shader stages in the pipeline there must not be any shader stage in the pipeline with a Xfb execution mode
*** whats a xfb

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-shaderMeshEnqueue-10187
* If the shaderMeshEnqueue feature is not enabled, shaders specified by pStages must not declare the ShaderEnqueueAMDX capability
https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-flags-10188
* If flags does not include VK_PIPELINE_CREATE_LIBRARY_BIT_KHR, shaders specified by pStages must not declare the ShaderEnqueueAMDX capability
*** my understanding is nabla strictly controls it's extensions, so this shouldnt be an issue

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07065
* If the pipeline requires pre-rasterization shader state, and includes a mesh shader, there must be no element of the
*  pDynamicStates member of pDynamicState set to VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, or VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE
*** this one seems the most relevant

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07066
* If the pipeline requires pre-rasterization shader state, and includes a mesh shader, there must be no element of the
*  pDynamicStates member of pDynamicState set to VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE, or VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07067
* If the pipeline requires pre-rasterization shader state, and includes a mesh shader, there must be no element of the pDynamicStates member of pDynamicState set to VK_DYNAMIC_STATE_VERTEX_INPUT_EXT

https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-07720
* If renderPass is VK_NULL_HANDLE, the pipeline is being created with pre-rasterization shader state, and
*  VkPipelineRenderingCreateInfo::viewMask is not 0, and multiviewMeshShader is not enabled, then pStages must not include a mesh shader


* theres 1 or 2 more about pipeline libraries, but im not going to worry about that
*/

namespace nbl::video
{

    class IGPUMeshPipeline : public IGPUPipeline<asset::IMeshPipeline<const IGPUPipelineLayout, const IGPURenderpass>>
    {
        using pipeline_t = asset::IMeshPipeline<const IGPUPipelineLayout, const IGPURenderpass>;

    public:
        struct SCreationParams final : public SPipelineCreationParams<const IGPUMeshPipeline>
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
                SSpecializationValidationResult retval = { .count = 0,.dataSize = 0 };

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576
                if (!renderpass || cached.subpassIx >= renderpass->getSubpassCount())
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
                if (!processSpecInfo(taskShader, hlsl::ShaderStage::ESS_TASK)) return {};
                if (!processSpecInfo(meshShader, hlsl::ShaderStage::ESS_MESH)) return {};
                if (!processSpecInfo(fragmentShader, hlsl::ShaderStage::ESS_FRAGMENT)) return {};

                if (!hasRequiredStages(stagePresence))
                    return {};

                //if (!vertexShader.shader) return {}; //i dont quite understand why this line was in IGPUGraphics. checking if the shader itself was made correctly?

                return retval;
            }

            inline core::bitflag<hlsl::ShaderStage> getRequiredSubgroupStages() const
            {

                core::bitflag<hlsl::ShaderStage> stages = {};
                auto processSpecInfo = [&](const SShaderSpecInfo& spec, hlsl::ShaderStage stage)
                    {
                        if (spec.shader && spec.requiredSubgroupSize >= SUBGROUP_SIZE::REQUIRE_4) {
                            stages |= stage;
                        }
                    };
                processSpecInfo(taskShader, hlsl::ESS_TASK);
                processSpecInfo(meshShader, hlsl::ESS_MESH);
                processSpecInfo(fragmentShader, hlsl::ESS_FRAGMENT);
                return stages;
            }

            inline core::bitflag<FLAGS>& getFlags() { return flags; }

            inline core::bitflag<FLAGS> getFlags() const { return flags; }

            const IGPUPipelineLayout* layout = nullptr;
            SShaderSpecInfo taskShader;
            SShaderSpecInfo meshShader;
            SShaderSpecInfo fragmentShader;
            SCachedCreationParams cached = {};
            renderpass_t* renderpass = nullptr;

            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;

            inline uint32_t getShaderCount() const
            {
                uint32_t count = 0; //count = 2 and only check task shader??
                count += (taskShader.shader != nullptr);
                count += (meshShader.shader != nullptr);
                count += (fragmentShader.shader != nullptr);
                return count;
            }
        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const {return m_flags;}

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

    protected:
        // not explicit?
        IGPUMeshPipeline(const SCreationParams& params) :
          IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice()), params.layout, params.cached, params.renderpass), m_flags(params.flags)
        {}
        virtual ~IGPUMeshPipeline() override = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
    };

}

#endif