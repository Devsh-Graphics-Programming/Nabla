#ifndef _NBL_VIDEO_I_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_I_QUERY_POOL_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{

class IQueryPool : public IBackendObject
{
    public:
            enum TYPE : uint16_t
            {
                OCCLUSION = 0x00000001,
                PIPELINE_STATISTICS = 0x00000002,
                TIMESTAMP = 0x00000004,
//                PERFORMANCE_QUERY = 0x00000008, // VK_KHR_performance_query // TODO: We don't support this fully yet -> needs Acquire/ReleaseProfilingLock + Counters Information report from physical device
                ACCELERATION_STRUCTURE_COMPACTED_SIZE = 0x00000010, // VK_KHR_acceleration_structure
                ACCELERATION_STRUCTURE_SERIALIZATION_SIZE = 0x00000020, // VK_KHR_acceleration_structure
                ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS = 0x00000040, // VK_KHR_ray_tracing_maintenance1
                ACCELERATION_STRUCTURE_SIZE = 0x00000080, // VK_KHR_ray_tracing_maintenance1
                COUNT = 8u,
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

            struct SCreationParams
            {
                uint32_t                                    queryCount;
                TYPE                                        queryType;
                // certain query types need some extra info
                union
                {
                    core::bitflag<PIPELINE_STATISTICS_FLAGS> pipelineStatisticsFlags;
                };
            };        
            inline const auto& getCreationParameters() const
            {
                return m_params;
            }

            enum RESULTS_FLAGS : uint8_t
            {
                NONE = 0,
                _64_BIT = 0x00000001,
                WAIT_BIT = 0x00000002,
                WITH_AVAILABILITY_BIT = 0x00000004,
                PARTIAL_BIT = 0x00000008,
            };
            static inline size_t calcQueryResultsSize(const SCreationParams& params, const size_t stride, const core::bitflag<RESULTS_FLAGS> flags)
            {
                if (params.queryCount==0u)
                    return 0ull;

                const size_t basicUnitSize = flags.hasFlags(RESULTS_FLAGS::_64_BIT) ? sizeof(uint64_t):sizeof(uint32_t);
                size_t singleQuerySize;
                switch (params.queryType)
                {
                    case IQueryPool::TYPE::OCCLUSION: [[fallthrough]];
                    case IQueryPool::TYPE::TIMESTAMP:[[fallthrough]];
                    case IQueryPool::TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE: [[fallthrough]];
                    case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: [[fallthrough]];
                    case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: [[fallthrough]];
                    case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SIZE:
                        singleQuerySize = basicUnitSize;
                        break;
                    case IQueryPool::TYPE::PIPELINE_STATISTICS:
                        singleQuerySize = basicUnitSize*hlsl::bitCount(static_cast<uint16_t>(params.pipelineStatisticsFlags.value));
                        break;
                    default:
                        return 0ull;
                }
                if (flags.hasFlags(IQueryPool::RESULTS_FLAGS::WITH_AVAILABILITY_BIT))
                    singleQuerySize += basicUnitSize;
                if (stride<singleQuerySize)
                    return 0ull;

                return stride*(params.queryCount-1u)+singleQuerySize;
            }

    protected:
            explicit inline IQueryPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SCreationParams& _params) 
                : IBackendObject(std::move(dev)), m_params(_params) {}

            SCreationParams m_params;
};

}

#endif