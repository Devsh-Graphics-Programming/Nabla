#include "nbl/asset/utils/ISPIRVDebloater.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl_spirv_cross/spirv.hpp"

#include "nbl/core/declarations.h"
#include "nbl/system/ILogger.h"


using namespace nbl::asset;

ISPIRVDebloater::ISPIRVDebloater()
{
    constexpr auto optimizationPasses = std::array{
        ISPIRVOptimizer::EOP_ELIM_DEAD_FUNCTIONS,
        ISPIRVOptimizer::EOP_ELIM_DEAD_VARIABLES,
        ISPIRVOptimizer::EOP_ELIM_DEAD_CONSTANTS,
        // TODO: remove unused type. no dedicated flag for type. maybe need agressive dce to remove type.
        ISPIRVOptimizer::EOP_TRIM_CAPABILITIES,
    };
    m_optimizer = core::make_smart_refctd_ptr<ISPIRVOptimizer>(std::span(optimizationPasses));
}

ISPIRVDebloater::Result ISPIRVDebloater::debloat(const  ICPUBuffer* spirvBuffer, std::span<const EntryPoint> entryPoints, system::logger_opt_ptr logger) const
{
    const auto* spirv = static_cast<const uint32_t*>(spirvBuffer->getPointer());
    const auto spirvDwordCount = spirvBuffer->getSize() / 4;

    auto getHlslShaderStage = [](spv::ExecutionModel executionModel) -> hlsl::ShaderStage
      {
        switch (executionModel)
        {
            case  spv::ExecutionModelVertex : return hlsl::ESS_VERTEX;
            case  spv::ExecutionModelTessellationControl : return hlsl::ESS_TESSELLATION_CONTROL;
            case  spv::ExecutionModelTessellationEvaluation : return hlsl::ESS_TESSELLATION_EVALUATION;
            case  spv::ExecutionModelGeometry : return hlsl::ESS_GEOMETRY;
            case  spv::ExecutionModelFragment : return hlsl::ESS_FRAGMENT;
            case  spv::ExecutionModelGLCompute : return hlsl::ESS_COMPUTE;
            case  spv::ExecutionModelTaskEXT : return hlsl::ESS_TASK;
            case  spv::ExecutionModelMeshEXT: return hlsl::ESS_MESH;
            case  spv::ExecutionModelRayGenerationKHR : return hlsl::ESS_RAYGEN;
            case  spv::ExecutionModelAnyHitKHR : return hlsl::ESS_ANY_HIT;
            case  spv::ExecutionModelClosestHitKHR : return hlsl::ESS_CLOSEST_HIT;
            case  spv::ExecutionModelMissKHR : return hlsl::ESS_MISS;
            case  spv::ExecutionModelIntersectionKHR : return hlsl::ESS_INTERSECTION;
            case  spv::ExecutionModelCallableKHR : return hlsl::ESS_CALLABLE;
            default:
            {
                assert(false);
                return hlsl::ESS_UNKNOWN;
            }
        }
      };
    static constexpr auto HEADER_SIZE = 5;

    core::vector<bool> foundEntryPoints(entryPoints.size(), false);
    std::vector<uint32_t> minimizedSpirv;
    bool needDebloat = false;

    auto offset = HEADER_SIZE;

    while (offset < spirvDwordCount) {
        const auto curOffset = offset;
        const auto instruction = spirv[curOffset];

        const auto length = instruction >> 16;
        const auto opcode = instruction & 0x0ffffu;

        offset += length;

        if (opcode == spv::OpEntryPoint)
        {
            const auto curExecutionModel = static_cast<spv::ExecutionModel>(spirv[curOffset + 1]);

            // TODO: check whether this reinterpret_cast is UB
            const auto curEntryPointName = std::string_view(reinterpret_cast<const char*>(spirv + curOffset + 3)); // entryPoint name is the third parameter

            auto findEntryPointIt = std::find(entryPoints.begin(), entryPoints.end(), 
              EntryPoint{
                .name = curEntryPointName,
                .shaderStage = getHlslShaderStage(curExecutionModel),
              });
            if (findEntryPointIt != entryPoints.end())
            {
                foundEntryPoints[std::distance(findEntryPointIt, entryPoints.begin())] = true;
            } else
            {
                if (needDebloat == false)
                {
                    minimizedSpirv.reserve(spirvDwordCount);
                    minimizedSpirv.insert(minimizedSpirv.end(), spirv, spirv + curOffset);
                    needDebloat = true;
                }
                continue;
            }
        }
        
        if (!needDebloat) continue;
        minimizedSpirv.insert(minimizedSpirv.end(), spirv + curOffset, spirv + offset);
    }

    const auto isAllEntryPointsFound = std::all_of(foundEntryPoints.begin(), foundEntryPoints.end(), std::identity{});

    if (!isAllEntryPointsFound || !needDebloat) {
        return Result {
            .isAllEntryPointsFound = isAllEntryPointsFound,
            .spirv = nullptr,
        };
    }

    return {
      .isAllEntryPointsFound = true,
      .spirv = m_optimizer->optimize(minimizedSpirv.data(), minimizedSpirv.size(), logger)
    };
    
}
