#include "nbl/asset/utils/ISPIRVEntryPointTrimmer.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl_spirv_cross/spirv.hpp"

#include "nbl/core/declarations.h"
#include "nbl/system/ILogger.h"
#include "spirv-tools/libspirv.hpp"

#include <compare>

using namespace nbl::asset;

static constexpr spv_target_env SPIRV_VERSION = spv_target_env::SPV_ENV_UNIVERSAL_1_6;

ISPIRVEntryPointTrimmer::ISPIRVEntryPointTrimmer()
{
    constexpr auto optimizationPasses = std::array{
        ISPIRVOptimizer::EOP_DEAD_BRANCH_ELIM,
        ISPIRVOptimizer::EOP_ELIM_DEAD_FUNCTIONS,
        ISPIRVOptimizer::EOP_ELIM_DEAD_VARIABLES,

        // This will remove spec constant as well based on this doc
        // https://github.com/KhronosGroup/SPIRV-Tools/blob/dec28643ed15f68a2bc95650de25e0a7486b564c/include/spirv-tools/optimizer.hpp#L349
        ISPIRVOptimizer::EOP_ELIM_DEAD_CONSTANTS,

        ISPIRVOptimizer::EOP_ELIM_DEAD_MEMBERS,

        // Based on experimentation, Aggresive DCE will remove unused type
        ISPIRVOptimizer::EOP_AGGRESSIVE_DCE,

        ISPIRVOptimizer::EOP_TRIM_CAPABILITIES,
    };
    m_optimizer = core::make_smart_refctd_ptr<ISPIRVOptimizer>(std::span(optimizationPasses));
}

// This is for debugging temporarily. will be reworked after finish testing 
static void printCapabilities(const uint32_t* spirv, uint32_t spirvDwordCount,nbl::system::logger_opt_ptr logger)
{
    spvtools::SpirvTools core(SPIRV_VERSION);
    std::string disassembly;
    core.Disassemble(spirv, spirvDwordCount, &disassembly, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
    std::stringstream ss(disassembly);
    std::string to;
    const auto stringsToFind = std::array{ "OpCapability", "= OpFunction","OpFunctionEnd", "OpSpecConstant", "=OpType"};
    while(std::getline(ss, to, '\n')){
      if (to.size() > 1 && to.back() == ',') continue;
      for (const auto& stringToFind: stringsToFind)
      {
        if (to.find(stringToFind) != std::string::npos)
        {
          logger.log("%s", nbl::system::ILogger::ELL_DEBUG, to.c_str());
        }
      }
    }
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
    spvtools::ValidatorOptions validatorOptions;
    // Nabla use Scalar block layout, we skip this validation to work around this and to save time
    validatorOptions.SetSkipBlockLayout(true);
    return core.Validate(binary, binarySize, validatorOptions);
}

ISPIRVEntryPointTrimmer::Result ISPIRVEntryPointTrimmer::trim(const  ICPUBuffer* spirvBuffer, const core::set<EntryPoint>& entryPoints, system::logger_opt_ptr logger) const
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

    auto foundEntryPoint = 0;

    const bool isInputSpirvValid  = validate(spirv, spirvDwordCount, logger);
    if (!isInputSpirvValid)
    {
        logger.log("SPIR-V is not valid", system::ILogger::ELL_ERROR);
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

    bool needtrim = false;
    auto offset = HEADER_SIZE;
    auto parse_instruction = [](uint32_t instruction) -> std::tuple<uint32_t, uint32_t>
    {
        const auto length = instruction >> 16;
        const auto opcode = instruction & 0x0ffffu;
        return { length, opcode };
    };

    // Keep in mind about this layout while reading all the code below: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#LogicalLayout

    // skip until entry point
    while (std::cmp_less(offset, spirvDwordCount)) {
        const auto instruction = spirv[offset];
        const auto [length, opcode] = parse_instruction(instruction);
        if (opcode == spv::OpEntryPoint) break;
        offset += length;
    }

    // handle entry points removal
    while (std::cmp_less(offset, spirvDwordCount)) {
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
            .stage = getHlslShaderStage(curExecutionModel),
        };

        if (entryPoint.stage == hlsl::ESS_UNKNOWN)
        {
            logger.log("Found entry point with unsupported execution model in SPIR-V", system::ILogger::ELL_ERROR);
            return Result{
                .spirv = nullptr,
                .isSuccess = false
            };
        }

        auto findEntryPointIt = entryPoints.find(entryPoint);
        if (findEntryPointIt != entryPoints.end())
        {
            foundEntryPoint += 1; // a valid spirv will have unique entry points, so this should works
        } else
        {
            if (needtrim == false)
            {
                minimizedSpirv.reserve(spirvDwordCount);
                minimizedSpirv.insert(minimizedSpirv.end(), spirv, spirv + curOffset);
                needtrim = true;
            }
            removedEntryPointIds.insert(curEntryPointId);
            continue;
        }
        if (!needtrim) continue;
        minimizedSpirv.insert(minimizedSpirv.end(), spirv + curOffset, spirv + offset);
    }

    const auto wereAllEntryPointsFound = std::cmp_equal(foundEntryPoint, entryPoints.size());
    if (!wereAllEntryPointsFound)
    {
        logger.log("Some entry point that is requested to be retained is not found in SPIR-V", system::ILogger::ELL_ERROR);
        return {
            .spirv = nullptr,
            .isSuccess = false,
        };
    }

    if (!needtrim)
    {
        return {
            .spirv = nullptr,
            .isSuccess = true,
        };
    }

    // handle execution model removal
    while (std::cmp_less(offset, spirvDwordCount))
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

    auto trimmedSpirv = m_optimizer->optimize(minimizedSpirv.data(), minimizedSpirv.size(), logger);

#ifdef _NBL_DEBUG
    logger.log("Before stripping capabilities:", nbl::system::ILogger::ELL_DEBUG);
    printCapabilities(spirv, spirvDwordCount, logger);
    logger.log("\n", nbl::system::ILogger::ELL_DEBUG);

    const auto* trimmedSpirvBuffer = static_cast<const uint32_t*>(trimmedSpirv->getPointer());
    const auto trimmedSpirvDwordCount = trimmedSpirv->getSize() / 4;
    logger.log("After stripping capabilities:", nbl::system::ILogger::ELL_DEBUG);
    printCapabilities(trimmedSpirvBuffer, trimmedSpirvDwordCount, logger);
    logger.log("\n", nbl::system::ILogger::ELL_DEBUG);
#endif

    return {
      .spirv = std::move(trimmedSpirv),
      .isSuccess = true,
    };
    
}

