#include "nbl/asset/ISPIRVOptimizer.h"

#include "spirv-tools/optimizer.hpp" 
#include "spirv-tools/libspirv.hpp"

#include "nbl/core/core.h"
#include "nbl/core/IReferenceCounted.h"
#include "os.h"

using namespace nbl::asset;

static constexpr spv_target_env SPIRV_VERSION = spv_target_env::SPV_ENV_UNIVERSAL_1_5;

nbl::core::smart_refctd_ptr<ICPUBuffer> ISPIRVOptimizer::optimize(const uint32_t* _spirv, uint32_t _dwordCount) const
{
    spvtools::Optimizer opt(SPIRV_VERSION);

    // lots of these taken from https://www.lunarg.com/wp-content/uploads/2020/05/SPIR-V-Shader-Legalization-and-Size-Reduction-Using-spirv-opt_v1.2.pdf
    opt .RegisterPass(spvtools::CreateMergeReturnPass())
        .RegisterPass(spvtools::CreateInlineExhaustivePass())
        .RegisterPass(spvtools::CreateEliminateDeadFunctionsPass())
        .RegisterPass(spvtools::CreateScalarReplacementPass())
        .RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(spvtools::CreateLocalSingleStoreElimPass())
        .RegisterPass(spvtools::CreateSimplificationPass())
        .RegisterPass(spvtools::CreateVectorDCEPass())
        .RegisterPass(spvtools::CreateDeadInsertElimPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        .RegisterPass(spvtools::CreateDeadBranchElimPass())
        .RegisterPass(spvtools::CreateBlockMergePass())
        .RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass())
        .RegisterPass(spvtools::CreateLocalSingleStoreElimPass())
        .RegisterPass(spvtools::CreateLocalMultiStoreElimPass())
        .RegisterPass(spvtools::CreateSimplificationPass())
        .RegisterPass(spvtools::CreateVectorDCEPass())
        .RegisterPass(spvtools::CreateDeadInsertElimPass())
        .RegisterPass(spvtools::CreateRedundancyEliminationPass())
        .RegisterPass(spvtools::CreateAggressiveDCEPass())
        //.RegisterPass(spvtools::CreateLoopInvariantCodeMotionPass()) //this completely breaks material compiler
        .RegisterPass(spvtools::CreateCCPPass())
        .RegisterPass(spvtools::CreateReduceLoadSizePass())
        .RegisterPass(spvtools::CreateStrengthReductionPass())
        .RegisterPass(spvtools::CreateIfConversionPass())
        ;

    auto msgConsumer = [](spv_message_level_t level, const char* src, const spv_position_t& pos, const char* msg)
    {
        using namespace std::string_literals;

        constexpr static ELOG_LEVEL lvl2lvl[6]{
            ELL_ERROR,
            ELL_ERROR,
            ELL_ERROR,
            ELL_WARNING,
            ELL_INFORMATION,
            ELL_DEBUG
        };
        const auto lvl = lvl2lvl[level];
        const std::string location = src + ":"s + std::to_string(pos.line) + ":" + std::to_string(pos.column);

        os::Printer::log(location, msg, lvl);
    };

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

nbl::core::smart_refctd_ptr<ICPUBuffer> ISPIRVOptimizer::optimize(const ICPUBuffer* _spirv) const
{
    const uint32_t* spirv = reinterpret_cast<const uint32_t*>(_spirv->getPointer());
    const uint32_t count = _spirv->getSize() / sizeof(uint32_t);

    return optimize(spirv, count);
}
