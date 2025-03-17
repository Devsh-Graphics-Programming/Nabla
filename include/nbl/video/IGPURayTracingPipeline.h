#ifndef _NBL_I_GPU_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IRayTracingPipeline.h"

#include "nbl/video/SPipelineCreationParams.h"


namespace nbl::video
{

class IGPURayTracingPipeline : public IBackendObject, public asset::IRayTracingPipeline<const IGPUPipelineLayout, const IGPUShader>
{
        using pipeline_t = asset::IRayTracingPipeline<const IGPUPipelineLayout,const IGPUShader>;

    public:

        struct SShaderGroupHandle
        {
          private:
            uint8_t data[video::SPhysicalDeviceLimits::ShaderGroupHandleSize];
        };
        static_assert(sizeof(SShaderGroupHandle) == video::SPhysicalDeviceLimits::ShaderGroupHandleSize);

        struct SHitGroupStackSize
        {
            uint16_t closestHit;
            uint16_t anyHit;
            uint16_t intersection;
        };

        struct SCreationParams final : pipeline_t::SCreationParams, SPipelineCreationParams<const IGPURayTracingPipeline>
        {

            inline SSpecializationValidationResult valid() const
            {
                if (!layout)
                  return {};

                SSpecializationValidationResult retval = {
                    .count=0,
                    .dataSize=0,
                };
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

            inline std::span<const IGPUShader::SSpecInfo> getShaders() const { return shaders; }

        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const { return m_flags; }

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

        virtual const SShaderGroupHandle& getRaygen() const = 0;
        virtual const SShaderGroupHandle& getMiss(uint32_t index) const = 0;
        virtual const SShaderGroupHandle& getHit(uint32_t index) const = 0;
        virtual const SShaderGroupHandle& getCallable(uint32_t index) const = 0;

        virtual uint16_t getRaygenStackSize() const = 0;
        virtual std::span<const uint16_t> getMissStackSizes() const = 0;
        virtual std::span<const SHitGroupStackSize> getHitStackSizes() const = 0;
        virtual std::span<const uint16_t> getCallableStackSizes() const = 0;
        virtual uint16_t getDefaultStackSize() const = 0;

    protected:
        IGPURayTracingPipeline(const SCreationParams& params) : IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice())),
            pipeline_t(params),
            m_flags(params.flags)
        {}

        virtual ~IGPURayTracingPipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};

}

#endif
