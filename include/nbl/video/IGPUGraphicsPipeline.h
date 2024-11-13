#ifndef _NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/asset/IGraphicsPipeline.h"

#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/video/IGPURenderpass.h"


namespace nbl::video
{

class IGPUGraphicsPipeline : public IBackendObject, public asset::IGraphicsPipeline<const IGPUPipelineLayout,const IGPUShader,const IGPURenderpass>
{
        using pipeline_t = asset::IGraphicsPipeline<const IGPUPipelineLayout,const IGPUShader,const IGPURenderpass>;

    public:
		struct SCreationParams final : pipeline_t::SCreationParams, SPipelineCreationParams<const IGPUGraphicsPipeline>
		{
            public:
            #define base_flag(F) static_cast<uint64_t>(pipeline_t::SCreationParams::FLAGS::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
                CAPTURE_STATISTICS = base_flag(CAPTURE_STATISTICS),
                CAPTURE_INTERNAL_REPRESENTATIONS = base_flag(CAPTURE_INTERNAL_REPRESENTATIONS),
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
                LINK_TIME_OPTIMIZATION = base_flag(LINK_TIME_OPTIMIZATION),
                RETAIN_LINK_TIME_OPTIMIZATION_INFO = base_flag(RETAIN_LINK_TIME_OPTIMIZATION_INFO)
            };
            #undef base_flag

            inline SSpecializationValidationResult valid() const
            {
                if (!layout)
                    return {};
                SSpecializationValidationResult retval = {.count=0,.dataSize=0};
                const bool valid = pipeline_t::SCreationParams::impl_valid([&retval](const IGPUShader::SSpecInfo& info)->bool
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

            inline std::span<const IGPUShader::SSpecInfo> getShaders() const {return shaders;}

            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
		};

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const {return m_flags;}

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

    protected:
        IGPUGraphicsPipeline(const SCreationParams& params) : IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice())),
            pipeline_t(params), m_flags(params.flags) {}
        virtual ~IGPUGraphicsPipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};

}

#endif