// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SCAN_ARITHMETIC_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCAN_ARITHMETIC_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{

template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2, uint16_t _ItemsPerInvocation>
struct ScanConfiguration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation = _ItemsPerInvocation;

    using arith_config_t = workgroup2::ArithmeticConfiguration<config_t::WorkgroupSizeLog2, config_t::SubgroupSizeLog2, config_t::ItemsPerInvocation>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedScratchElementCount = arith_config_t::SharedScratchElementCount;
};

namespace impl
{

template<typename T>    // only uint32_t or uint64_t for now?
struct Constants
{
    NBL_CONSTEXPR_STATIC_INLINE T NOT_READY = 0;
    NBL_CONSTEXPR_STATIC_INLINE T LOCAL_COUNT = T(0x1u) << (sizeof(T)*8-2);
    NBL_CONSTEXPR_STATIC_INLINE T GLOBAL_COUNT = T(0x1u) << (sizeof(T)*8-1);
    NBL_CONSTEXPR_STATIC_INLINE T STATUS_MASK = LOCAL_COUNT | GLOBAL_COUNT;
};

template<class Config, class BinOp, bool ForwardProgressGuarantees, class device_capabilities>
struct reduce
{
    using scalar_t = typename BinOp::type_t;
    using constants_t = Constants<scalar_t>;
    using config_t = Config;
    using arith_config_t = typename Config::arith_config_t;
    using workgroup_reduce_t = workgroup2::reduction<arith_config_t, BinOp, device_capabilities>;
    using binop_t = BinOp;

    template<class DataAccessor, class ScratchAccessor>
    scalar_t __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) sharedMemScratchAccessor)
    {
        const scalar_t localReduction = workgroup_reduce_t::__call<DataAccessor, ScratchAccessor>(dataAccessor, sharedMemScratchAccessor);
        bda::__ptr<T> scratch = dataAccessor.getScratchPtr();   // scratch data should be at least T[NumWorkgroups]

        const bool lastInvocation = (workgroup::SubgroupContiguousIndex() == WorkgroupSize-1);
        if (lastInvocation)
        {
            bda::__ref<T> scratchId = (scratch + glsl::gl_WorkgroupID()).deref();
            spirv::atomicUMax(scratchId.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsReleaseMask, localReduction|constants_t::LOCAL_COUNT);
        }

        scalar_t prefix = scalar_t(0);
        // decoupled lookback
        if (ForwardProgressGuarantees)
        {
            if (lastInvocation) // don't make whole block work and do busy stuff
            {
                for (uint32_t prevID = glsl::gl_WorkgroupID()-1; prevID > 0u; prevID--)
                {
                    scalar_t value = scalar_t(0);
                    {
                        // spin until something is ready
                        while (value == constants_t::NOT_READY)
                        {
                            bda::__ref<uint32_t,4> scratchPrev = (scratch-1).deref();
                            value = spirv::atomicLoad(scratchPrev.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask);
                        }
                    }
                    prefix += value & (~constants_t::STATUS_MASK);

                    // last was actually a global sum, we have the prefix, we can quit
                    if (value & constants_t::GLOBAL_COUNT)
                        break;
                }
            }
            prefix = workgroup::Broadcast(prefix, sharedMemScratchAccessor, WorkgroupSize-1);
        }

        binop_t binop;
        scalar_t globalReduction = binop(prefix,localReduction);
        if (lastInvocation)
        {
            bda::__ref<uint32_t,4> scratchId = (scratch + glsl::gl_WorkgroupID()).deref();
            spirv::atomicUMax(scratchId.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsReleaseMask, globalReduction|constants_t::GLOBAL_COUNT);
        }

        // get last item from scratch
        uint32_t lastWorkgroup = glsl::gl_NumWorkgroups() - 1;
        bda::__ref<uint32_t,4> scratchLast = (scratch + lastWorkgroup).deref();
        uint32_t value;
        {
            // wait until last workgroup does reduction
            while (value & constants_t::GLOBAL_COUNT)
            {
                value = spirv::atomicLoad(scratchLast.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask);
            }
        }
        return value & (~constants_t::STATUS_MASK);
    }
}

}

}
}
}

#endif
