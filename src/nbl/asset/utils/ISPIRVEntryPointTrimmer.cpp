#include "nbl/asset/utils/ISPIRVEntryPointTrimmer.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl_spirv_cross/spirv.hpp"

#include "nbl/core/declarations.h"
#include "nbl/system/ILogger.h"
#include "spirv-tools/libspirv.hpp"

using namespace nbl::asset;

// Why are we validating Universal instead of a Vulkan environment?
// Trimming works on generic SPIR-V before the Vulkan backend chooses its environment.
static constexpr spv_target_env SPIRV_VALIDATION_ENV = spv_target_env::SPV_ENV_UNIVERSAL_1_6;

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
    spvtools::SpirvTools core(SPIRV_VALIDATION_ENV);
    core.SetMessageConsumer(msgConsumer);
    spvtools::ValidatorOptions validatorOptions;
    // Nabla use Scalar block layout, we skip this validation to work around this and to save time
    validatorOptions.SetSkipBlockLayout(true);
    return core.Validate(binary, binarySize, validatorOptions);
}

static nbl::core::blake3_hash_t getContentHash(const ICPUBuffer* spirvBuffer)
{
    auto contentHash = spirvBuffer->getContentHash();
    if (contentHash == ICPUBuffer::INVALID_HASH)
        contentHash = spirvBuffer->computeContentHash();
    return contentHash;
}

bool ISPIRVEntryPointTrimmer::ensureValidated(const ICPUBuffer* spirvBuffer, system::logger_opt_ptr logger) const
{
    const auto contentHash = getContentHash(spirvBuffer);

    {
        std::lock_guard lock(m_validationCacheMutex);
        if (m_validatedSpirvHashes.contains(contentHash))
            return true;
    }

    const auto* spirv = static_cast<const uint32_t*>(spirvBuffer->getPointer());
    const auto spirvDwordCount = spirvBuffer->getSize() / sizeof(uint32_t);
    if (!validate(spirv, spirvDwordCount, logger))
        return false;

    {
        std::lock_guard lock(m_validationCacheMutex);
        m_validatedSpirvHashes.emplace(contentHash);
    }

    return true;
}

void ISPIRVEntryPointTrimmer::markValidated(const ICPUBuffer* spirvBuffer) const
{
    std::lock_guard lock(m_validationCacheMutex);
    m_validatedSpirvHashes.emplace(getContentHash(spirvBuffer));
}

ISPIRVEntryPointTrimmer::Result ISPIRVEntryPointTrimmer::trim(const  ICPUBuffer* spirvBuffer, const core::set<EntryPoint>& entryPoints, system::logger_opt_ptr logger) const
{
    const auto* spirv = static_cast<const uint32_t*>(spirvBuffer->getPointer());
    const auto spirvDwordCount = spirvBuffer->getSize() / sizeof(uint32_t);

    if (entryPoints.empty())
    {
       logger.log("Cannot retain zero multiple entry points!", system::ILogger::ELL_ERROR);
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

    if (!ensureValidated(spirvBuffer, logger))
    {
        logger.log("SPIR-V is not valid", system::ILogger::ELL_ERROR);
        return Result{
            .spirv = nullptr,
            .isSuccess = false,
        };
    }

    {
        auto probeOffset = HEADER_SIZE;
        auto totalEntryPoints = 0u;
        auto matchingEntryPoints = 0u;
        auto validFastPath = (spirvDwordCount >= HEADER_SIZE);

        while (validFastPath && probeOffset < spirvDwordCount)
        {
            const auto instruction = spirv[probeOffset];
            const auto [length, opcode] = parse_instruction(instruction);
            if (length == 0u || (probeOffset + length) > spirvDwordCount)
            {
                validFastPath = false;
                break;
            }
            if (opcode == spv::OpEntryPoint)
                break;
            probeOffset += length;
        }

        while (validFastPath && probeOffset < spirvDwordCount)
        {
            const auto curOffset = probeOffset;
            const auto instruction = spirv[curOffset];
            const auto [length, opcode] = parse_instruction(instruction);
            if (length == 0u || (probeOffset + length) > spirvDwordCount)
            {
                validFastPath = false;
                break;
            }
            if (opcode != spv::OpEntryPoint)
                break;
            probeOffset += length;
            ++totalEntryPoints;

            const auto curExecutionModel = static_cast<spv::ExecutionModel>(spirv[curOffset + 1]);
            const auto curEntryPointName = std::string_view(reinterpret_cast<const char*>(spirv + curOffset + 3));
            const auto entryPoint = EntryPoint{
                .name = curEntryPointName,
                .stage = getHlslShaderStage(curExecutionModel),
            };
            if (entryPoint.stage == hlsl::ESS_UNKNOWN)
            {
                validFastPath = false;
                break;
            }
            if (entryPoints.contains(entryPoint))
                ++matchingEntryPoints;
        }

        if (validFastPath && totalEntryPoints == entryPoints.size() && matchingEntryPoints == entryPoints.size())
        {
            return {
                .spirv = nullptr,
                .isSuccess = true,
            };
        }
    }

    auto foundEntryPoint = 0;

    // Keep in mind about this layout while reading all the code below: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#LogicalLayout

    // skip until entry point
    while (offset < spirvDwordCount) {
        const auto instruction = spirv[offset];
        const auto [length, opcode] = parse_instruction(instruction);
        if (opcode == spv::OpEntryPoint) break;
        offset += length;
    }

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

    const auto wereAllEntryPointsFound = foundEntryPoint == entryPoints.size();
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

    auto trimmedSpirv = m_optimizer->optimize(minimizedSpirv.data(), minimizedSpirv.size(), logger);
    if (!trimmedSpirv)
    {
        logger.log("Failed to optimize trimmed SPIR-V", system::ILogger::ELL_ERROR);
        return {
            .spirv = nullptr,
            .isSuccess = false,
        };
    }

    trimmedSpirv->setContentHash(trimmedSpirv->computeContentHash());
    if (!ensureValidated(trimmedSpirv.get(), logger))
    {
        logger.log("Trimmed SPIR-V is not valid", system::ILogger::ELL_ERROR);
        return {
            .spirv = nullptr,
            .isSuccess = false,
        };
    }

    return {
      .spirv = std::move(trimmedSpirv),
      .isSuccess = true,
    };
    
}

