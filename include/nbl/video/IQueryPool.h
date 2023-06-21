#ifndef _NBL_VIDEO_I_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_I_QUERY_POOL_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{

class IQueryPool : public core::IReferenceCounted, public IBackendObject
{
    public:
            enum TYPE : uint8_t
            {
                OCCLUSION = 0x00000001,
                PIPELINE_STATISTICS = 0x00000002,
                TIMESTAMP = 0x00000004,
                PERFORMANCE_QUERY = 0x00000008, // VK_KHR_performance_query // TODO: We don't support this fully yet -> needs Acquire/ReleaseProfilingLock + Counters Information report from physical device
                ACCELERATION_STRUCTURE_COMPACTED_SIZE = 0x00000010, // VK_KHR_acceleration_structure
                ACCELERATION_STRUCTURE_SERIALIZATION_SIZE = 0x00000020, // VK_KHR_acceleration_structure
                COUNT = 6u,
            };

            enum class PIPELINE_STATISTICS_FLAGS : uint16_t
            {
                NONE = 0,
                INPUT_ASSEMBLY_VERTICES_BIT = 0x00000001,
                INPUT_ASSEMBLY_PRIMITIVES_BIT = 0x00000002,
                VERTEX_SHADER_INVOCATIONS_BIT = 0x00000004,
                GEOMETRY_SHADER_INVOCATIONS_BIT = 0x00000008,
                GEOMETRY_SHADER_PRIMITIVES_BIT = 0x00000010,
                CLIPPING_INVOCATIONS_BIT = 0x00000020,
                CLIPPING_PRIMITIVES_BIT = 0x00000040,
                FRAGMENT_SHADER_INVOCATIONS_BIT = 0x00000080,
                TESSELLATION_CONTROL_SHADER_PATCHES_BIT = 0x00000100,
                TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT = 0x00000200,
                COMPUTE_SHADER_INVOCATIONS_BIT = 0x00000400,
            };

            enum RESULTS_FLAGS : uint8_t
            {
                EQRF_NONE = 0,
                EQRF_64_BIT = 0x00000001,
                EQRF_WAIT_BIT = 0x00000002,
                EQRF_WITH_AVAILABILITY_BIT = 0x00000004,
                EQRF_PARTIAL_BIT = 0x00000008,
            };

            struct SCreationParams
            {
                uint32_t                                    queryCount;
                core::bitflag<PIPELINE_STATISTICS_FLAGS>    pipelineStatisticsFlags; // only when the queryType is EQT_PIPELINE_STATISTICS
                TYPE                                        queryType;
            };

    public:
            explicit inline IQueryPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _params) 
                : IBackendObject(std::move(dev)), params(std::move(_params)) {}
        
            inline const auto& getCreationParameters() const
            {
                return params;
            }

    protected:
            SCreationParams params;
};

}

#endif