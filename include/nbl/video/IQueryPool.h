#ifndef _NBL_VIDEO_I_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_I_QUERY_POOL_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{

class IQueryPool : public core::IReferenceCounted, public IBackendObject
{
    
public:
        enum E_QUERY_TYPE
        {
            EQT_OCCLUSION = 0,
            EQT_PIPELINE_STATISTICS = 1,
            EQT_TIMESTAMP = 2,
            EQT_TRANSFORM_FEEDBACK_STREAM_EXT = 1000028004, // VK_EXT_transform_feedback
            EQT_PERFORMANCE_QUERY = 1000116000, // VK_KHR_performance_query
            EQT_ACCELERATION_STRUCTURE_COMPACTED_SIZE = 1000150000, // VK_KHR_acceleration_structure
            EQT_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE = 1000150001, // VK_KHR_acceleration_structure
        };

        enum E_PIPELINE_STATISTICS_FLAGS
        {
            EPSF_INPUT_ASSEMBLY_VERTICES_BIT = 0x00000001,
            EPSF_INPUT_ASSEMBLY_PRIMITIVES_BIT = 0x00000002,
            EPSF_VERTEX_SHADER_INVOCATIONS_BIT = 0x00000004,
            EPSF_GEOMETRY_SHADER_INVOCATIONS_BIT = 0x00000008,
            EPSF_GEOMETRY_SHADER_PRIMITIVES_BIT = 0x00000010,
            EPSF_CLIPPING_INVOCATIONS_BIT = 0x00000020,
            EPSF_CLIPPING_PRIMITIVES_BIT = 0x00000040,
            EPSF_FRAGMENT_SHADER_INVOCATIONS_BIT = 0x00000080,
            EPSF_TESSELLATION_CONTROL_SHADER_PATCHES_BIT = 0x00000100,
            EPSF_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT = 0x00000200,
            EPSF_COMPUTE_SHADER_INVOCATIONS_BIT = 0x00000400,
        };

        enum E_QUERY_RESULTS_FLAGS
        {
            EQRF_64_BIT = 0x00000001,
            EQRF_WAIT_BIT = 0x00000002,
            EQRF_WITH_AVAILABILITY_BIT = 0x00000004,
            EQRF_PARTIAL_BIT = 0x00000008,
        };

        enum E_QUERY_CONTROL_FLAGS : uint32_t
        {
            EQCF_PRECISE_BIT = 0x01
        };

        struct SCreationParams
        {
            E_QUERY_TYPE                    queryType;
            uint32_t                        queryCount;
            E_PIPELINE_STATISTICS_FLAGS     pipelineStatisticsFlags; // only when the queryType is EQT_PIPELINE_STATISTICS
        };

public:
        explicit IQueryPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _params) 
            : IBackendObject(std::move(dev)) 
            , params(std::move(_params))
        {}
        
protected:
        SCreationParams params;
};

}

#endif