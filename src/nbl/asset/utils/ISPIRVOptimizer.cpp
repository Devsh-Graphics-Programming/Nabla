#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "spirv-tools/optimizer.hpp" 
#include "spirv-tools/libspirv.hpp"

#include "nbl/core/declarations.h"
#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ILogger.h"

using namespace nbl::asset;

static constexpr spv_target_env SPIRV_VERSION = spv_target_env::SPV_ENV_UNIVERSAL_1_5;

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
    create_pass_f_t create_pass_f[EOP_COUNT]{
        &spvtools::CreateMergeReturnPass,
        &spvtools::CreateInlineExhaustivePass,
        &spvtools::CreateEliminateDeadFunctionsPass,
        CreateScalarReplacementPass,
        &spvtools::CreateLocalSingleBlockLoadStoreElimPass,
        &spvtools::CreateLocalSingleStoreElimPass,
        &spvtools::CreateSimplificationPass,
        &spvtools::CreateVectorDCEPass,
        &spvtools::CreateDeadInsertElimPass,
        //&spvtools::CreateAggressiveDCEPass,
        &spvtools::CreateDeadBranchElimPass,
        &spvtools::CreateBlockMergePass,
        &spvtools::CreateLocalMultiStoreElimPass,
        &spvtools::CreateRedundancyEliminationPass,
        &spvtools::CreateLoopInvariantCodeMotionPass,
        &spvtools::CreateCCPPass,
        CreateReduceLoadSizePass,
        &spvtools::CreateStrengthReductionPass,
        &spvtools::CreateIfConversionPass
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
        const std::string location = src + ":"s + std::to_string(pos.line) + ":" + std::to_string(pos.column);

        logger.log(location, lvl, msg);
    };

    spvtools::Optimizer opt(SPIRV_VERSION);

    for (E_OPTIMIZER_PASS pass : m_passes)
        opt.RegisterPass(create_pass_f[pass]());

    opt.SetMessageConsumer(msgConsumer);

    std::vector<uint32_t> optimized;
    opt.Run(_spirv, _dwordCount, &optimized);

    const uint32_t resultBytesize = optimized.size() * sizeof(uint32_t);
    if (!resultBytesize)
        return nullptr;

    auto result = core::make_smart_refctd_ptr<ICPUBuffer>(resultBytesize);
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
