#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "spirv-tools/optimizer.hpp"

#include "nbl/core/declarations.h"
#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ILogger.h"

using namespace nbl::asset;

static constexpr spv_target_env SPIRV_VERSION = spv_target_env::SPV_ENV_UNIVERSAL_1_6;

nbl::core::smart_refctd_ptr<ICPUBuffer> ISPIRVOptimizer::optimize(const uint32_t* _spirv, uint32_t _dwordCount, system::logger_opt_ptr logger) const
{
    //https://www.lunarg.com/wp-content/uploads/2020/05/SPIR-V-Shader-Legalization-and-Size-Reduction-Using-spirv-opt_v1.2.pdf

    auto CreateScalarReplacementPass = [] {
        return spvtools::CreateScalarReplacementPass();
    };

    auto CreateReduceLoadSizePass = [] {
        return spvtools::CreateReduceLoadSizePass();
    };

    using create_pass_f_t = spvtools::Optimizer::PassToken(*)();
    auto getSpirvOptimizerPass = [&](E_OPTIMIZER_PASS pass) -> create_pass_f_t
    {
        switch (pass)
        {
        case EOP_MERGE_RETURN: return &spvtools::CreateMergeReturnPass;
        case EOP_INLINE: return &spvtools::CreateInlineExhaustivePass;
        case EOP_ELIM_DEAD_FUNCTIONS: return &spvtools::CreateEliminateDeadFunctionsPass;
        case EOP_ELIM_DEAD_VARIABLES: return &spvtools::CreateDeadVariableEliminationPass;
        case EOP_ELIM_DEAD_CONSTANTS: return &spvtools::CreateEliminateDeadConstantPass;
        case EOP_ELIM_DEAD_MEMBERS: return &spvtools::CreateEliminateDeadMembersPass;
        case EOP_SCALAR_REPLACEMENT: return CreateScalarReplacementPass;
        case EOP_LOCAL_SINGLE_BLOCK_LOAD_STORE_ELIM: return &spvtools::CreateLocalSingleBlockLoadStoreElimPass;
        case EOP_LOCAL_SINGLE_STORE_ELIM: return &spvtools::CreateLocalSingleStoreElimPass;
        case EOP_SIMPLIFICATION: return &spvtools::CreateSimplificationPass;
        case EOP_VECTOR_DCE: return &spvtools::CreateVectorDCEPass;
        case EOP_DEAD_INSERT_ELIM: return &spvtools::CreateDeadInsertElimPass;
        case EOP_DEAD_BRANCH_ELIM: return &spvtools::CreateDeadBranchElimPass;
        case EOP_BLOCK_MERGE: return &spvtools::CreateBlockMergePass;
        case EOP_LOCAL_MULTI_STORE_ELIM: return &spvtools::CreateLocalMultiStoreElimPass;
        case EOP_REDUNDANCY_ELIM: return &spvtools::CreateRedundancyEliminationPass;
        case EOP_LOOP_INVARIANT_CODE_MOTION: return &spvtools::CreateLoopInvariantCodeMotionPass;
        case EOP_CCP: return &spvtools::CreateCCPPass;
        case EOP_REDUCE_LOAD_SIZE: return CreateReduceLoadSizePass;
        case EOP_STRENGTH_REDUCTION: return &spvtools::CreateStrengthReductionPass;
        case EOP_IF_CONVERSION: return &spvtools::CreateIfConversionPass;
        case EOP_STRIP_DEBUG_INFO: return &spvtools::CreateStripDebugInfoPass;
        case EOP_TRIM_CAPABILITIES: return &spvtools::CreateTrimCapabilitiesPass;
        case EOP_AGGRESSIVE_DCE: return &spvtools::CreateAggressiveDCEPass;
        case EOP_REMOVE_UNUSED_INTERFACE_VARIABLES: return &spvtools::CreateRemoveUnusedInterfaceVariablesPass;
        case EOP_ELIMINATE_DEAD_INPUT_COMPONENTS_SAFE: return &spvtools::CreateEliminateDeadInputComponentsSafePass;
        default:
            return nullptr;
        }
    };

    create_pass_f_t create_pass_f[EOP_COUNT]{
        &spvtools::CreateMergeReturnPass,
        &spvtools::CreateInlineExhaustivePass,
        &spvtools::CreateEliminateDeadFunctionsPass,
        &spvtools::CreateDeadVariableEliminationPass,
        &spvtools::CreateEliminateDeadConstantPass,
        &spvtools::CreateEliminateDeadMembersPass,
        CreateScalarReplacementPass,
        &spvtools::CreateLocalSingleBlockLoadStoreElimPass,
        &spvtools::CreateLocalSingleStoreElimPass,
        &spvtools::CreateSimplificationPass,
        &spvtools::CreateVectorDCEPass,
        &spvtools::CreateDeadInsertElimPass,
        &spvtools::CreateDeadBranchElimPass,
        &spvtools::CreateBlockMergePass,
        &spvtools::CreateLocalMultiStoreElimPass,
        &spvtools::CreateRedundancyEliminationPass,
        &spvtools::CreateLoopInvariantCodeMotionPass,
        &spvtools::CreateCCPPass,
        CreateReduceLoadSizePass,
        &spvtools::CreateStrengthReductionPass,
        &spvtools::CreateIfConversionPass,
        &spvtools::CreateStripDebugInfoPass,
        &spvtools::CreateTrimCapabilitiesPass,
        &spvtools::CreateAggressiveDCEPass,
        &spvtools::CreateRemoveUnusedInterfaceVariablesPass,
        &spvtools::CreateEliminateDeadInputComponentsSafePass,
    };


    auto msgConsumer = [&logger](spv_message_level_t level, const char* src, const spv_position_t& pos, const char* msg)
    {
        using namespace std::string_literals;


        constexpr static system::ILogger::E_LOG_LEVEL lvl2lvl[6]{
            system::ILogger::ELL_ERROR,
            system::ILogger::ELL_ERROR,
            system::ILogger::ELL_ERROR,
            system::ILogger::ELL_WARNING,
            system::ILogger::ELL_INFO,
            system::ILogger::ELL_DEBUG
        };
        const auto lvl = lvl2lvl[level];
        std::string location;
        if (src)
            location = src + ":"s + std::to_string(pos.line) + ":" + std::to_string(pos.column);
        else
            location = "";

        logger.log(location, lvl, msg);
    };

    spvtools::Optimizer opt(SPIRV_VERSION);

    for (E_OPTIMIZER_PASS pass : m_passes) {
        if (getSpirvOptimizerPass(pass) != nullptr)
        {
            opt.RegisterPass(create_pass_f[pass]());
        } else
        {
            logger.log("Optimizer pass is unknown or not supported!", system::ILogger::ELL_WARNING);
        }
    }

    opt.SetMessageConsumer(msgConsumer);

    std::vector<uint32_t> optimized;
    opt.Run(_spirv, _dwordCount, &optimized);

    const uint32_t resultBytesize = optimized.size() * sizeof(uint32_t);
    if (!resultBytesize)
        return nullptr;

    auto result = ICPUBuffer::create({ resultBytesize });
    memcpy(result->getPointer(), optimized.data(), resultBytesize);

    return result;
}

nbl::core::smart_refctd_ptr<ICPUBuffer> ISPIRVOptimizer::optimize(const ICPUBuffer* _spirv, system::logger_opt_ptr logger) const
{
    const uint32_t* spirv = reinterpret_cast<const uint32_t*>(_spirv->getPointer());
    const uint32_t count = _spirv->getSize() / sizeof(uint32_t);

    return optimize(spirv, count, logger);
}

const std::span<const ISPIRVOptimizer::E_OPTIMIZER_PASS> nbl::asset::ISPIRVOptimizer::getPasses() const
{
    return std::span{m_passes};
}
