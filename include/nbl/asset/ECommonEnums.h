#ifndef _NBL_ASSET_E_COMMON_ENUMS_H_INCLUDED_
#define _NBL_ASSET_E_COMMON_ENUMS_H_INCLUDED_

#include "nbl/core/declarations.h"

namespace nbl::asset
{

// would be in a common asset::IPipeline if it existed
enum E_PIPELINE_BIND_POINT : uint8_t
{
    EPBP_GRAPHICS = 0,
    EPBP_COMPUTE,

    EPBP_COUNT
};

// here because acceleration structures need them too
enum E_INDEX_TYPE : uint8_t
{
    EIT_16BIT = 0,
    EIT_32BIT,
    EIT_UNKNOWN
};

// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-pipeline-stages-order
enum class PIPELINE_STAGE_FLAGS : uint32_t
{
    NONE = 0,
    HOST_BIT = 0x1<<0,
    COPY_BIT = 0x1<<1,
    CLEAR_BIT = 0x1<<2,
    //    MICROMAP_BUILD_BIT = 0x1<<3,
    ACCELERATION_STRUCTURE_COPY_BIT = 0x1<<4,
    ACCELERATION_STRUCTURE_BUILD_BIT = 0x1<<5,
    // this is for the indirect of indirects commands in the commandbuffer
    COMMAND_PREPROCESS_BIT = 0x1<<6,
    // while the spec says this stage exists on its own, I'd think it needs to ofc execute before any indirect stuff
    CONDITIONAL_RENDERING_BIT = 0x1<<7,
    // begin of any pipeline
    DISPATCH_INDIRECT_COMMAND_BIT = 0x1<<8,
    // compute pipeline only has a single unique stage
    COMPUTE_SHADER_BIT = 0x1<<9,
    // primitive pipeline
    INDEX_INPUT_BIT = 0x1<<10,
    VERTEX_ATTRIBUTE_INPUT_BIT = 0x1<<11,
    VERTEX_INPUT_BITS = INDEX_INPUT_BIT|VERTEX_ATTRIBUTE_INPUT_BIT,
    VERTEX_SHADER_BIT = 0x1<<12,
    TESSELLATION_CONTROL_SHADER_BIT = 0x1<<13,
    TESSELLATION_EVALUATION_SHADER_BIT = 0x1<<14,
    GEOMETRY_SHADER_BIT = 0x1<<15,
    //! We do not expose Transform Feedback
    // TODO: mesh pipeline
//    TASK_SHADER_BIT = 0x1<<16,
//    MESH_SHADER_BIT = 0x1<<17,
    // 
    PRE_RASTERIZATION_SHADERS_BITS = VERTEX_SHADER_BIT|TESSELLATION_CONTROL_SHADER_BIT|TESSELLATION_EVALUATION_SHADER_BIT|GEOMETRY_SHADER_BIT/*|TASK_SHADER_BIT|MESH_SHADER_BIT*/,
    // similar to shading rate but affects raster rate as well
    FRAGMENT_DENSITY_PROCESS_BIT = 0x1<<18,
    // rasterization but not 1:1 with a single pixel
    SHADING_RATE_ATTACHMENT_BIT = 0x1<<19,
    // framebuffer space (for a single sample, relevant for BY_REGION deps)
    EARLY_FRAGMENT_TESTS_BIT = 0x1<<20,
    FRAGMENT_SHADER_BIT = 0x1<<21,
    LATE_FRAGMENT_TESTS_BIT = 0x1<<22,
    COLOR_ATTACHMENT_OUTPUT_BIT = 0x1<<23,
    FRAMEBUFFER_SPACE_BITS = EARLY_FRAGMENT_TESTS_BIT|FRAGMENT_SHADER_BIT|LATE_FRAGMENT_TESTS_BIT|COLOR_ATTACHMENT_OUTPUT_BIT,
    // one more left
    RAY_TRACING_SHADER_BIT = 0x1<<24,
    // framebuffer attachment stuff
    RESOLVE_BIT = 0x1<<25,
    BLIT_BIT = 0x1<<26,
    // TODOs
//  VIDEO_DECODE_BIT = 0x1<<27,
//  VIDEO_ENCODE_BIT = 0x1<<28,
//  OPTICAL_FLOW_BIT = 0x1<<29,
    // special
    ALL_TRANSFER_BITS = COPY_BIT|ACCELERATION_STRUCTURE_COPY_BIT|CLEAR_BIT|RESOLVE_BIT|BLIT_BIT,
    ALL_GRAPHICS_BITS = CONDITIONAL_RENDERING_BIT|DISPATCH_INDIRECT_COMMAND_BIT|
        VERTEX_INPUT_BITS|VERTEX_SHADER_BIT|TESSELLATION_CONTROL_SHADER_BIT|TESSELLATION_EVALUATION_SHADER_BIT|GEOMETRY_SHADER_BIT|
 //       TASK_SHADER_BIT|MESH_SHADER_BIT|
        FRAGMENT_DENSITY_PROCESS_BIT|SHADING_RATE_ATTACHMENT_BIT|FRAMEBUFFER_SPACE_BITS,
    ALL_COMMANDS_BITS = ~HOST_BIT
};
NBL_ENUM_ADD_BITWISE_OPERATORS(PIPELINE_STAGE_FLAGS)

enum class ACCESS_FLAGS : uint32_t
{
    NONE = 0,
    HOST_READ_BIT = 0x1u<<0,
    HOST_WRITE_BIT = 0x1u<<1,
    TRANSFER_READ_BIT = 0x1u<<2,
    TRANSFER_WRITE_BIT = 0x1u<<3,
//    MICROMAP_READ_BIT = 0x1u<<4,
//    MICROMAP_WRITE_BIT = 0x1u<<5,
    ACCELERATION_STRUCTURE_READ_BIT = 0x1u<<6,
    ACCELERATION_STRUCTURE_WRITE_BIT = 0x1u<<7,
    // indirect commands
    COMMAND_PREPROCESS_READ_BIT = 0x1u<<8,
    COMMAND_PREPROCESS_WRITE_BIT = 0x1u<<9,
    CONDITIONAL_RENDERING_READ_BIT = 0x1u<<10,
    INDIRECT_COMMAND_READ_BIT = 0x1u<<11,
    // common to all stages
    UNIFORM_READ_BIT = 0x1u<<12,
    SAMPLED_READ_BIT = 0x1u<<13,
    STORAGE_READ_BIT = 0x1u<<14,
    STORAGE_WRITE_BIT = 0x1u<<15,
    // vertex only
    INDEX_READ_BIT = 0x1u<<16,
    VERTEX_ATTRIBUTE_READ_BIT = 0x1u<<17,
    //! We do not expose Transform Feedback
    // fragment only
    FRAGMENT_DENSITY_MAP_READ_BIT = 0x1u<<18,
    SHADING_RATE_ATTACHMENT_READ_BIT = 0x1u<<19,
    DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x1u<<20,
    DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x1u<<21,
    INPUT_ATTACHMENT_READ_BIT = 0x1u<<22,
    COLOR_ATTACHMENT_READ_BIT = 0x1u<<23,
    //! We will never support `VK_EXT_blend_operation_advanced` without coherent blending, or even at all
    //! because you can implement whatever blend you want with Fragment Shader Interlock and friends
    COLOR_ATTACHMENT_WRITE_BIT = 0x1u<<24,
    //
    SHADER_BINDING_TABLE_READ_BIT = 0x1u<<25,
    // TODOs
//    VIDEO_DECODE_READ_BIT=0x1u<<26,
//    VIDEO_DECODE_WRITE_BIT=0x1u<<27,
//    VIDEO_ENCODE_READ_BIT=0x1u<<28,
//    VIDEO_ENCODE_WRITE_BIT=0x1u<<29,
//    OPTICAL_FLOW_READ_BIT=0x1u<<30,
//    OPTICAL_FLOW_WRITE_BIT=0x1u<<31,
    // special
    SHADER_READ_BITS = ACCELERATION_STRUCTURE_READ_BIT|UNIFORM_READ_BIT|SAMPLED_READ_BIT|STORAGE_READ_BIT|INPUT_ATTACHMENT_READ_BIT|SHADER_BINDING_TABLE_READ_BIT, // I'll deviate from Vulkan spec here a bit
    SHADER_WRITE_BITS = STORAGE_WRITE_BIT,
    MEMORY_READ_BITS = HOST_READ_BIT|TRANSFER_READ_BIT/*|MICROMAP_READ_BIT*/|ACCELERATION_STRUCTURE_READ_BIT|COMMAND_PREPROCESS_READ_BIT|
        CONDITIONAL_RENDERING_READ_BIT|INDIRECT_COMMAND_READ_BIT|
        SHADER_READ_BITS|INDEX_READ_BIT|VERTEX_ATTRIBUTE_READ_BIT|
        FRAGMENT_DENSITY_MAP_READ_BIT|SHADING_RATE_ATTACHMENT_READ_BIT|INPUT_ATTACHMENT_READ_BIT|DEPTH_STENCIL_ATTACHMENT_READ_BIT|COLOR_ATTACHMENT_READ_BIT|
        SHADER_BINDING_TABLE_READ_BIT,
        //VIDEO_DECODE_READ_BIT|VIDEO_ENCODE_READ_BIT|OPTICAL_FLOW_READ_BIT,
    MEMORY_WRITE_BITS = HOST_WRITE_BIT|TRANSFER_WRITE_BIT/*|MICROMAP_WRITE_BIT*/|ACCELERATION_STRUCTURE_WRITE_BIT|COMMAND_PREPROCESS_WRITE_BIT|
        SHADER_WRITE_BITS|DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|COLOR_ATTACHMENT_WRITE_BIT,
        //VIDEO_DECODE_WRITE_BIT|VIDEO_ENCODE_WRITE_BIT|OPTICAL_FLOW_WRITE_BIT,
};
NBL_ENUM_ADD_BITWISE_OPERATORS(ACCESS_FLAGS)

struct SMemoryBarrier
{
    // TODO: pack these up into just `src` and `dst` with another struct
    core::bitflag<PIPELINE_STAGE_FLAGS> srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
    core::bitflag<ACCESS_FLAGS> srcAccessMask = ACCESS_FLAGS::NONE;
    core::bitflag<PIPELINE_STAGE_FLAGS> dstStageMask = PIPELINE_STAGE_FLAGS::NONE;
    core::bitflag<ACCESS_FLAGS> dstAccessMask = ACCESS_FLAGS::NONE;

    auto operator<=>(const SMemoryBarrier&) const = default;

    // Make the immediately previous barrier after this one
    inline SMemoryBarrier prevBarrier(const core::bitflag<PIPELINE_STAGE_FLAGS> prevStageMask, const core::bitflag<ACCESS_FLAGS> prevAccessMask) const
    {
        return {
            .srcStageMask = prevStageMask,
            .srcAccessMask = prevAccessMask,
            .dstStageMask = srcStageMask,
            .dstAccessMask = srcAccessMask
        };
    }
    // Make the immediately previous barrier if you know the penultimate, as in one before the last (previous)
    inline SMemoryBarrier prevBarrier(const SMemoryBarrier& penultimate) const
    {
        return prevBarrier(penultimate.dstStageMask,penultimate.dstAccessMask);
    }
    // Make the immediately next barrier after this one
    inline SMemoryBarrier nextBarrier(const core::bitflag<PIPELINE_STAGE_FLAGS> nextStageMask, const core::bitflag<ACCESS_FLAGS> nextAccessMask) const
    {
        return {
            .srcStageMask = dstStageMask,
            .srcAccessMask = dstAccessMask,
            .dstStageMask = nextStageMask,
            .dstAccessMask = nextAccessMask
        };
    }
    // Make the immediately next barrier, if you know the barrier thats after the immediately next one
    inline SMemoryBarrier nextBarrier(const SMemoryBarrier& twoAhead) const
    {
        return prevBarrier(twoAhead.srcStageMask,twoAhead.srcAccessMask);
    }
};

inline core::bitflag<ACCESS_FLAGS> allAccessesFromStages(core::bitflag<PIPELINE_STAGE_FLAGS> stages)
{
    struct PerStageAccesses
    {
        public:
            constexpr PerStageAccesses()
            {
                init(PIPELINE_STAGE_FLAGS::HOST_BIT,ACCESS_FLAGS::HOST_READ_BIT|ACCESS_FLAGS::HOST_WRITE_BIT);

                constexpr auto TransferRW = ACCESS_FLAGS::TRANSFER_READ_BIT|ACCESS_FLAGS::TRANSFER_WRITE_BIT;
                init(PIPELINE_STAGE_FLAGS::COPY_BIT,TransferRW);
                init(PIPELINE_STAGE_FLAGS::CLEAR_BIT,ACCESS_FLAGS::TRANSFER_WRITE_BIT);

                constexpr auto MicromapRead = ACCESS_FLAGS::SHADER_READ_BITS;//|ACCESS_FLAGS::MICROMAP_READ_BIT;
//                init(PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT,MicromapRead|ACCESS_FLAGS::MICROMAP_WRITE_BIT); // can micromaps be built indirectly?
                
                constexpr auto AccelerationStructureRW = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT|ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT;
                init(PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT,TransferRW|AccelerationStructureRW);
                init(PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT,ACCESS_FLAGS::INDIRECT_COMMAND_READ_BIT|MicromapRead|AccelerationStructureRW);

                init(PIPELINE_STAGE_FLAGS::COMMAND_PREPROCESS_BIT,ACCESS_FLAGS::COMMAND_PREPROCESS_READ_BIT|ACCESS_FLAGS::COMMAND_PREPROCESS_WRITE_BIT);
                init(PIPELINE_STAGE_FLAGS::CONDITIONAL_RENDERING_BIT,ACCESS_FLAGS::CONDITIONAL_RENDERING_READ_BIT);
                init(PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT,ACCESS_FLAGS::INDIRECT_COMMAND_READ_BIT);

                constexpr auto ShaderRW = ACCESS_FLAGS::SHADER_READ_BITS|ACCESS_FLAGS::SHADER_WRITE_BITS;
                constexpr auto AllShaderStagesRW = ShaderRW^(ACCESS_FLAGS::INPUT_ATTACHMENT_READ_BIT|ACCESS_FLAGS::SHADER_BINDING_TABLE_READ_BIT);
                init(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,AllShaderStagesRW);
                init(PIPELINE_STAGE_FLAGS::INDEX_INPUT_BIT,ACCESS_FLAGS::INDEX_READ_BIT);
                init(PIPELINE_STAGE_FLAGS::VERTEX_ATTRIBUTE_INPUT_BIT,ACCESS_FLAGS::VERTEX_ATTRIBUTE_READ_BIT);
                init(PIPELINE_STAGE_FLAGS::VERTEX_SHADER_BIT,AllShaderStagesRW);
                init(PIPELINE_STAGE_FLAGS::TESSELLATION_CONTROL_SHADER_BIT,AllShaderStagesRW);
                init(PIPELINE_STAGE_FLAGS::TESSELLATION_EVALUATION_SHADER_BIT,AllShaderStagesRW);
                init(PIPELINE_STAGE_FLAGS::GEOMETRY_SHADER_BIT,AllShaderStagesRW);
//                init(PIPELINE_STAGE_FLAGS::TASK_SHADER_BIT,AllShaderStagesRW);
//                init(PIPELINE_STAGE_FLAGS::MESH_SHADER_BIT,AllShaderStagesRW);
                init(PIPELINE_STAGE_FLAGS::FRAGMENT_DENSITY_PROCESS_BIT,ACCESS_FLAGS::FRAGMENT_DENSITY_MAP_READ_BIT);
                init(PIPELINE_STAGE_FLAGS::SHADING_RATE_ATTACHMENT_BIT,ACCESS_FLAGS::SHADING_RATE_ATTACHMENT_READ_BIT);
                constexpr auto DepthStencilRW = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT|ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                init(PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,DepthStencilRW);
                init(PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,AllShaderStagesRW|ACCESS_FLAGS::INPUT_ATTACHMENT_READ_BIT);
                init(PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,DepthStencilRW);
                init(PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,ACCESS_FLAGS::COLOR_ATTACHMENT_READ_BIT|ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT);

                init(PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,AllShaderStagesRW|ACCESS_FLAGS::SHADER_BINDING_TABLE_READ_BIT);

                init(PIPELINE_STAGE_FLAGS::RESOLVE_BIT,TransferRW);
                init(PIPELINE_STAGE_FLAGS::BLIT_BIT,TransferRW);

//                init(PIPELINE_STAGE_FLAGS::VIDEO_DECODE_BIT,ACCESS_FLAGS::VIDEO_DECODE_READ_BIT|ACCESS_FLAGS::VIDEO_DECODE_WRITE_BIT);
//                init(PIPELINE_STAGE_FLAGS::VIDEO_ENCODE_BIT,ACCESS_FLAGS::VIDEO_ENCODE_READ_BIT|ACCESS_FLAGS::VIDEO_ENCODE_WRITE_BIT);
//                init(PIPELINE_STAGE_FLAGS::OPTICAL_FLOW_BIT,ACCESS_FLAGS::OPTICAL_FLOW_READ_BIT|ACCESS_FLAGS::OPTICAL_FLOW_WRITE_BIT);
            }
            constexpr const auto& operator[](const size_t ix) const {return data[ix];}

        private:
            constexpr static uint8_t findLSB(size_t val)
            {
                for (size_t ix=0ull; ix<sizeof(size_t)*8; ix++)
                if ((0x1ull<<ix)&val)
                    return ix;
                return ~0u;
            }
            constexpr void init(PIPELINE_STAGE_FLAGS stageFlag, ACCESS_FLAGS accessFlags)
            {
                const auto bitIx = findLSB(static_cast<size_t>(stageFlag));
                data[bitIx] = accessFlags;
            }

            ACCESS_FLAGS data[32] = {};
    };
    constexpr PerStageAccesses bitToAccess = {};

    core::bitflag<ACCESS_FLAGS> retval = ACCESS_FLAGS::NONE;
    while (bool(stages.value))
    {
        const auto bitIx = hlsl::findLSB(stages);
        retval |= bitToAccess[bitIx];
        stages ^= static_cast<PIPELINE_STAGE_FLAGS>(0x1u<<bitIx);
    }

    return retval;
}

inline core::bitflag<PIPELINE_STAGE_FLAGS> allStagesFromAccesses(core::bitflag<ACCESS_FLAGS> accesses)
{
    struct PerAccessStages
    {
        public:
            constexpr PerAccessStages()
            {
                init(ACCESS_FLAGS::HOST_READ_BIT,PIPELINE_STAGE_FLAGS::HOST_BIT);
                init(ACCESS_FLAGS::HOST_WRITE_BIT,PIPELINE_STAGE_FLAGS::HOST_BIT);

                init(ACCESS_FLAGS::TRANSFER_READ_BIT,PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS^PIPELINE_STAGE_FLAGS::CLEAR_BIT);
                init(ACCESS_FLAGS::TRANSFER_WRITE_BIT,PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);

                constexpr auto MicromapAccelerationStructureBuilds = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;//|PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT;
//                init(ACCESS_FLAGS::MICROMAP_READ_BIT,MicromapAccelerationStructureBuilds);
//                init(ACCESS_FLAGS::MICROMAP_WRITE_BIT,PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT);
                
                constexpr auto AllShaders = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT|PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS|PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT|PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;
                constexpr auto AccelerationStructureOperations = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT|PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT;
                init(ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT,AccelerationStructureOperations|AllShaders);
                init(ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT,AccelerationStructureOperations);

                init(ACCESS_FLAGS::COMMAND_PREPROCESS_READ_BIT,PIPELINE_STAGE_FLAGS::COMMAND_PREPROCESS_BIT);
                init(ACCESS_FLAGS::COMMAND_PREPROCESS_WRITE_BIT,PIPELINE_STAGE_FLAGS::COMMAND_PREPROCESS_BIT);
                init(ACCESS_FLAGS::CONDITIONAL_RENDERING_READ_BIT,PIPELINE_STAGE_FLAGS::CONDITIONAL_RENDERING_BIT);
                init(ACCESS_FLAGS::INDIRECT_COMMAND_READ_BIT,PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT|PIPELINE_STAGE_FLAGS::DISPATCH_INDIRECT_COMMAND_BIT);

                init(ACCESS_FLAGS::UNIFORM_READ_BIT,AllShaders);
                init(ACCESS_FLAGS::SAMPLED_READ_BIT,AllShaders);//|PIPELINE_STAGE_FLAGS::MICROMAP_BUILD_BIT);
                init(ACCESS_FLAGS::STORAGE_READ_BIT,AllShaders|MicromapAccelerationStructureBuilds);
                init(ACCESS_FLAGS::STORAGE_WRITE_BIT,AllShaders);

                init(ACCESS_FLAGS::INDEX_READ_BIT,PIPELINE_STAGE_FLAGS::INDEX_INPUT_BIT);
                init(ACCESS_FLAGS::VERTEX_ATTRIBUTE_READ_BIT,PIPELINE_STAGE_FLAGS::VERTEX_ATTRIBUTE_INPUT_BIT);

                init(ACCESS_FLAGS::FRAGMENT_DENSITY_MAP_READ_BIT,PIPELINE_STAGE_FLAGS::FRAGMENT_DENSITY_PROCESS_BIT);
                init(ACCESS_FLAGS::SHADING_RATE_ATTACHMENT_READ_BIT,PIPELINE_STAGE_FLAGS::SHADING_RATE_ATTACHMENT_BIT);
                constexpr auto FragmentTests = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT;
                init(ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT,FragmentTests);
                init(ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,FragmentTests);
                init(ACCESS_FLAGS::INPUT_ATTACHMENT_READ_BIT,PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT);
                init(ACCESS_FLAGS::COLOR_ATTACHMENT_READ_BIT,PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT);
                init(ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT);

                init(ACCESS_FLAGS::SHADER_BINDING_TABLE_READ_BIT,PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT);

//                init(ACCESS_FLAGS::VIDEO_DECODE_READ_BIT,PIPELINE_STAGE_FLAGS::VIDEO_DECODE_BIT);
//                init(ACCESS_FLAGS::VIDEO_DECODE_WRITE_BIT,PIPELINE_STAGE_FLAGS::VIDEO_DECODE_BIT);
//                init(ACCESS_FLAGS::VIDEO_ENCODE_READ_BIT,PIPELINE_STAGE_FLAGS::VIDEO_ENCODE_BIT);
//                init(ACCESS_FLAGS::VIDEO_ENCODE_WRITE_BIT,PIPELINE_STAGE_FLAGS::VIDEO_ENCODE_BIT);
//                init(ACCESS_FLAGS::OPTICAL_FLOW_READ_BIT,PIPELINE_STAGE_FLAGS::OPTICAL_FLOW_BIT);
//                init(ACCESS_FLAGS::OPTICAL_FLOW_WRITE_BIT,PIPELINE_STAGE_FLAGS::OPTICAL_FLOW_BIT);
            }
            constexpr const auto& operator[](const size_t ix) const {return data[ix];}

        private:
            constexpr static uint8_t findLSB(size_t val)
            {
                for (size_t ix=0ull; ix<sizeof(size_t)*8; ix++)
                if ((0x1ull<<ix)&val)
                    return ix;
                return ~0u;
            }
            constexpr void init(ACCESS_FLAGS accessFlags, PIPELINE_STAGE_FLAGS stageFlags)
            {
                const auto bitIx = findLSB(static_cast<size_t>(accessFlags));
                data[bitIx] = stageFlags;
            }

            PIPELINE_STAGE_FLAGS data[32] = {};
    };
    constexpr PerAccessStages bitToStage = {};

    core::bitflag<PIPELINE_STAGE_FLAGS> retval = PIPELINE_STAGE_FLAGS::NONE;
    while (bool(accesses.value))
    {
        const auto bitIx = hlsl::findLSB(accesses);
        retval |= bitToStage[bitIx];
        accesses ^= static_cast<ACCESS_FLAGS>(0x1u<<bitIx);
    }

    return retval;
}

}

#endif