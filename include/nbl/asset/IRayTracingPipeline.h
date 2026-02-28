#ifndef _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/IPipeline.h"

#include <span>
#include <bit>
#include <type_traits>

namespace nbl::asset
{

class IRayTracingPipelineBase : public virtual core::IReferenceCounted
{
    public:
        #define base_flag(F) static_cast<uint64_t>(IPipelineBase::FLAGS::F)
        enum class CreationFlags : uint64_t
        {
            NONE = base_flag(NONE),
            // there's a bit of a problem, as the ICPUCompute and Graphics pipelines don't care about flags, because the following 4 flags
            DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
            ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
            FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
            EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
            // don't matter for ICPU Pipelines, we'd really need to have these separate from `base_flag` and use the `IRayTracingPipelineBase::CreationFlags` for the ICPU creation params only
            SKIP_BUILT_IN_PRIMITIVES = 1<<12,
            SKIP_AABBS = 1<<13,
            NO_NULL_ANY_HIT_SHADERS = 1<<14,
            NO_NULL_CLOSEST_HIT_SHADERS = 1<<15,
            NO_NULL_MISS_SHADERS = 1<<16,
            NO_NULL_INTERSECTION_SHADERS = 1<<17,
            ALLOW_MOTION = 1<<20,
            CAPTURE_STATISTICS = base_flag(CAPTURE_STATISTICS),
            CAPTURE_INTERNAL_REPRESENTATIONS = base_flag(CAPTURE_INTERNAL_REPRESENTATIONS),
        };
        #undef base_flag

        struct SCachedCreationParams final
        {
            core::bitflag<CreationFlags> flags = CreationFlags::NONE;
            uint32_t maxRecursionDepth : 6 = 0;
            uint32_t dynamicStackSize : 1 = false;
        };
};

template<typename PipelineLayoutType, typename BufferType>
class IRayTracingPipeline : public IPipeline<PipelineLayoutType>, public IRayTracingPipelineBase
{
    public:
        struct SShaderBindingTable
        {
            inline bool valid(const core::bitflag<IRayTracingPipelineBase::CreationFlags> flags) const
            {
                return valid(flags,[](const std::string_view, auto... args)->void{});
            }
            template<typename Callback>
            inline bool valid(const core::bitflag<IRayTracingPipelineBase::CreationFlags> flags, Callback&& cb) const
            {
                using create_flag_e = IRayTracingPipelineBase::CreationFlags;
                // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03696
                // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03697
                // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03512
                const auto shouldHaveHitGroup = flags & (core::bitflag(create_flag_e::NO_NULL_ANY_HIT_SHADERS) | create_flag_e::NO_NULL_CLOSEST_HIT_SHADERS | create_flag_e::NO_NULL_INTERSECTION_SHADERS);
                if (shouldHaveHitGroup && !hit.range.buffer)
                {
                    cb("bound pipeline indicates that traceRays command should have hit group, but SRayTracingSBT::hit::range::buffer is null!");
                    return false;
                }

                // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03511
                const auto shouldHaveMissGroup = flags & create_flag_e::NO_NULL_MISS_SHADERS;
                if (shouldHaveMissGroup && !miss.range.buffer)
                {
                    cb("bound pipeline indicates that traceRays command should have miss group, but SRayTracingSBT::hit::range::buffer is null!");
                    return false;
                }

                auto invalidBufferRegion = [&cb](const SStridedRange<const BufferType>& stRange, const char* groupName) -> bool
                {
                    const auto& range = stRange.range;
                    const auto* const buffer = range.buffer.get();
                    if (!buffer)
                        return false;

                    if (!range.isValid())
                    {
                        cb("%s buffer range is not valid!",groupName);
                        return false;
                    }

                    return false;
                };

                if (invalidBufferRegion({.range=raygen},"Raygen Group")) return false;
                if (invalidBufferRegion(miss,"Miss groups")) return false;
                if (invalidBufferRegion(hit,"Hit groups")) return false;
                if (invalidBufferRegion(callable,"Callable groups")) return false;

                return true;
            }

            asset::SBufferRange<BufferType> raygen = {};
            asset::SStridedRange<BufferType> miss = {}, hit = {}, callable = {};
        };

        inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }

    protected:
        explicit inline IRayTracingPipeline(PipelineLayoutType* layout, const SCachedCreationParams& cachedParams) :
            IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<PipelineLayoutType>(layout)), m_params(cachedParams) {}

        SCachedCreationParams m_params;

};

}

#endif
