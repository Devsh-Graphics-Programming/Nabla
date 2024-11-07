#ifndef _NBL_ASSET_I_SPIRV_OPTIMIZER_H_INCLUDED_
#define _NBL_ASSET_I_SPIRV_OPTIMIZER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUBuffer.h"

#include "nbl/system/ILogger.h"

namespace nbl::asset
{

class ISPIRVOptimizer final : public core::IReferenceCounted
{
    public:
        enum E_OPTIMIZER_PASS
        {
            EOP_MERGE_RETURN,
            EOP_INLINE,
            EOP_ELIM_DEAD_FUNCTIONS,
            EOP_SCALAR_REPLACEMENT,
            EOP_LOCAL_SINGLE_BLOCK_LOAD_STORE_ELIM,
            EOP_LOCAL_SINGLE_STORE_ELIM,
            EOP_SIMPLIFICATION,
            EOP_VECTOR_DCE,
            EOP_DEAD_INSERT_ELIM,
            EOP_AGGRESSIVE_DCE,
            EOP_DEAD_BRANCH_ELIM,
            EOP_BLOCK_MERGE,
            EOP_LOCAL_MULTI_STORE_ELIM,
            EOP_REDUNDANCY_ELIM,
            EOP_LOOP_INVARIANT_CODE_MOTION,
            EOP_CCP,
            EOP_REDUCE_LOAD_SIZE,
            EOP_STRENGTH_REDUCTION,
            EOP_IF_CONVERSION,

            EOP_COUNT
        };

        ISPIRVOptimizer(std::initializer_list<E_OPTIMIZER_PASS> _passes) : m_passes(std::move(_passes)) {}

        core::smart_refctd_ptr<ICPUBuffer> optimize(const uint32_t* _spirv, uint32_t _dwordCount, system::logger_opt_ptr logger) const;
        core::smart_refctd_ptr<ICPUBuffer> optimize(const ICPUBuffer* _spirv, system::logger_opt_ptr logger) const;
        const std::span<const E_OPTIMIZER_PASS> getPasses() const;

    protected:
        const core::vector<E_OPTIMIZER_PASS> m_passes;
};

}

#endif