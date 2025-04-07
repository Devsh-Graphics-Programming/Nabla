#include "nbl/asset/utils/ISPIRVDebloater.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl_spirv_cross/spirv.hpp"

#include "nbl/core/declarations.h"
#include "nbl/system/ILogger.h"


using namespace nbl::asset;

static constexpr spv_target_env SPIRV_VERSION = spv_target_env::SPV_ENV_UNIVERSAL_1_6;

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

static bool validate(const uint32_t* binary, uint32_t binarySize, nbl::system::logger_opt_ptr logger)
{
    auto msgConsumer = [&logger](spv_message_level_t level, const char* src, const spv_position_t& pos, const char* msg)
    {
        using namespace std::string_literals;


        constexpr static nbl::system::ILogger::E_LOG_LEVEL lvl2lvl[6]{
            nbl::system::ILogger::ELL_ERROR,
            nbl::system::ILogger::ELL_ERROR,
            nbl::system::ILogger::ELL_ERROR,
            nbl::system::ILogger::ELL_WARNING,
            nbl::system::ILogger::ELL_INFO,
            nbl::system::ILogger::ELL_DEBUG
        };
        const auto lvl = lvl2lvl[level];
        std::string location;
        if (src)
            location = src + ":"s + std::to_string(pos.line) + ":" + std::to_string(pos.column);
        else
            location = "";

        logger.log(location, lvl, msg);
    };
    spvtools::SpirvTools core(SPIRV_VERSION);
    core.SetMessageConsumer(msgConsumer);
    return core.Validate(binary, binarySize);
}

ISPIRVDebloater::Result ISPIRVDebloater::debloat(const  ICPUBuffer* spirvBuffer, std::span<const EntryPoint> entryPoints, system::logger_opt_ptr logger) const
{
    const auto* spirv = static_cast<const uint32_t*>(spirvBuffer->getPointer());
    const auto spirvDwordCount = spirvBuffer->getSize() / 4;

    if (entryPoints.empty())
    {
       logger.log("Cannot retain zero multiple entry points!", system::ILogger::ELL_ERROR);
       return Result{
          nullptr,
          false
       };
    }

    // We will remove found entry point one by one. We set all entry points as unfound initially
    core::set<EntryPoint> unfoundEntryPoints(entryPoints.begin(), entryPoints.end());
    for (const auto& entryPoint : entryPoints)
    {
       if (unfoundEntryPoints.find(entryPoint) != unfoundEntryPoints.end())
       {
           logger.log("Cannot retain multiple entry points with the same name and stage!", system::ILogger::ELL_ERROR);
           return Result{
              nullptr,
              false
           };
       }
       unfoundEntryPoints.insert(entryPoint);
    }

    const bool isInputSpirvValid  = validate(spirv, spirvDwordCount, logger);
    if (!isInputSpirvValid)
    {
        return Result{
            nullptr,
            false
        };
    }

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
                return hlsl::ESS_UNKNOWN;
            }
        }
      };

    static constexpr auto HEADER_SIZE = 5;

    std::vector<uint32_t> minimizedSpirv;
    core::unordered_set<uint32_t> removedEntryPointIds;

    bool needDebloat = false;
    auto offset = HEADER_SIZE;
    auto parse_instruction = [](uint32_t instruction) -> std::tuple<uint32_t, uint32_t>
    {
        const auto length = instruction >> 16;
        const auto opcode = instruction & 0x0ffffu;
        return { length, opcode };
    };

    // skip until entry point
    while (offset < spirvDwordCount) {
        const auto instruction = spirv[offset];
        const auto [length, opcode] = parse_instruction(instruction);
        if (opcode == spv::OpEntryPoint) break;
        offset += length;
    }

    const auto wereAllEntryPointsFound = unfoundEntryPoints.empty();
    // handle entry points removal
    while (offset < spirvDwordCount) {
        const auto curOffset = offset;
        const auto instruction = spirv[curOffset];
        const auto [length, opcode] = parse_instruction(instruction);
        if (opcode != spv::OpEntryPoint) break;
        offset += length;

        const auto curExecutionModel = static_cast<spv::ExecutionModel>(spirv[curOffset + 1]);
        const auto curEntryPointId = spirv[curOffset + 2];
        const auto curEntryPointName = std::string_view(reinterpret_cast<const char*>(spirv + curOffset + 3));

        const auto entryPoint = EntryPoint{
            .name = curEntryPointName,
            .shaderStage = getHlslShaderStage(curExecutionModel),
        };

        if (entryPoint.shaderStage == hlsl::ESS_UNKNOWN)
        {
            return Result{
                .spirv = nullptr,
                .isValid = false
            };
        }

        auto findEntryPointIt = unfoundEntryPoints.find(entryPoint);
        if (findEntryPointIt != unfoundEntryPoints.end())
        {
            unfoundEntryPoints.erase(findEntryPointIt);
        } else
        {
            if (needDebloat == false)
            {
                minimizedSpirv.reserve(spirvDwordCount);
                minimizedSpirv.insert(minimizedSpirv.end(), spirv, spirv + curOffset);
                needDebloat = true;
            }
            removedEntryPointIds.insert(curEntryPointId);
            continue;
        }
        if (!needDebloat) continue;
        minimizedSpirv.insert(minimizedSpirv.end(), spirv + curOffset, spirv + offset);
    }

    if (!needDebloat)
    {
        logger.log("Found entry point with unsupported execution model in spirv");
        return {
            .spirv = nullptr,
            .isValid = wereAllEntryPointsFound,
        };
    }

    // handle execution model removal
    while (offset < spirvDwordCount)
    {
        const auto curOffset = offset;
        const auto instruction = spirv[curOffset];
        const auto [length, opcode] = parse_instruction(instruction);
        if (opcode != spv::OpExecutionMode && opcode != spv::OpExecutionModeId) break;
        offset += length;
        const auto entryPointId = static_cast<spv::ExecutionModel>(spirv[curOffset + 1]);
        if (removedEntryPointIds.contains(entryPointId))
        {
            continue;
        }
        minimizedSpirv.insert(minimizedSpirv.end(), spirv + curOffset, spirv + offset);
    }

    minimizedSpirv.insert(minimizedSpirv.end(), spirv + offset, spirv + spirvDwordCount);

    assert(validate(minimizedSpirv.data(), minimizedSpirv.size(), logger));

    auto debloatedSpirv = m_optimizer->optimize(minimizedSpirv.data(), minimizedSpirv.size(), logger);

    return {
      .spirv = std::move(debloatedSpirv),
      .isValid = true,
    };
    
}
