// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SCAN_ARITHMETIC_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCAN_ARITHMETIC_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
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

    using arith_config_t = workgroup2::ArithmeticConfiguration<WorkgroupSizeLog2, SubgroupSizeLog2, ItemsPerInvocation>;
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
        const scalar_t localReduction = workgroup_reduce_t::template __call<DataAccessor, ScratchAccessor>(dataAccessor, sharedMemScratchAccessor);
        bda::__ptr<scalar_t> scratch = dataAccessor.getScratchPtr();   // scratch data should be at least T[NumWorkgroups]

        const bool lastInvocation = (workgroup::SubgroupContiguousIndex() == Config::WorkgroupSize-1);
        if (lastInvocation)
        {
            bda::__ref<scalar_t> scratchId = (scratch + glsl::gl_WorkGroupID().x).deref();
            spirv::atomicUMax(scratchId.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsReleaseMask, localReduction|constants_t::LOCAL_COUNT);
        }

        // NOTE: just for testing, remove when done
        // sharedMemScratchAccessor.workgroupExecutionAndMemoryBarrier();
        // uint32_t prev = glsl::gl_WorkGroupID().x==0 ? 0 : glsl::gl_WorkGroupID().x-1;
        // scalar_t testVal = constants_t::NOT_READY;
        // if (lastInvocation)
        //     while (testVal == constants_t::NOT_READY)
        //         testVal = spirv::atomicIAdd((scratch + prev).deref().__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask, 0u);
        // sharedMemScratchAccessor.workgroupExecutionAndMemoryBarrier();
        // testVal = workgroup::Broadcast(testVal, sharedMemScratchAccessor, Config::WorkgroupSize-1);
        // return testVal;

        binop_t binop;
        scalar_t prefix = scalar_t(0);
        // decoupled lookback
        if (ForwardProgressGuarantees)
        {
            if (lastInvocation) // don't make whole block work and do busy stuff
            {
                // for (uint32_t prevID = glsl::gl_WorkGroupID().x-1; prevID >= 0u; prevID--)   // won't run properly this way for some reason, results in device lost
                for (uint32_t i = 1; i <= glsl::gl_WorkGroupID().x; i++)
                {
                    const uint32_t prevID = glsl::gl_WorkGroupID().x-i;
                    scalar_t value = constants_t::NOT_READY;
                    {
                        // spin until something is ready
                        while (value == constants_t::NOT_READY)
                        {
                            bda::__ref<scalar_t> scratchPrev = (scratch + prevID).deref();
                            // value = spirv::atomicLoad(scratchPrev.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask);
                            value = spirv::atomicIAdd(scratchPrev.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask, 0u);
                        }
                    }
                    prefix = binop(value & (~constants_t::STATUS_MASK), prefix);

                    // last was actually a global sum, we have the prefix, we can quit
                    if (value & constants_t::GLOBAL_COUNT)
                        break;
                }
            }
            prefix = workgroup::Broadcast(prefix, sharedMemScratchAccessor, Config::WorkgroupSize-1);
        }
        else
        {
            // for (uint32_t prevID = glsl::gl_WorkGroupID().x-1; prevID >= 0u; prevID--)
            for (uint32_t i = 1; i <= glsl::gl_WorkGroupID().x; i++)
            {
                const uint32_t prevID = glsl::gl_WorkGroupID().x-i;
                scalar_t value = scalar_t(0);
                if (lastInvocation)
                {
                    bda::__ref<scalar_t> scratchPrev = (scratch + prevID).deref();
                    // value = spirv::atomicLoad(scratchPrev.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask);
                    value = spirv::atomicIAdd(scratchPrev.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask, 0u);
                }
                value = workgroup::Broadcast(value, sharedMemScratchAccessor, Config::WorkgroupSize-1);

                if (value & constants_t::STATUS_MASK)
                {
                    prefix = binop(value & (~constants_t::STATUS_MASK), prefix);

                    if (value & constants_t::GLOBAL_COUNT)
                        break;
                }
                else    // can't wait/spin, have to do it ourselves
                {
                    sharedMemScratchAccessor.workgroupExecutionAndMemoryBarrier();

                    DataAccessor prevDataAccessor = DataAccessor::create(prevID);
                    prevDataAccessor.begin();   // prepare data accessor if needed (e.g. preload)
                    const scalar_t prevReduction = workgroup_reduce_t::template __call<DataAccessor, ScratchAccessor>(prevDataAccessor, sharedMemScratchAccessor);

                    // if DoAndRaceStore, stores in place of prev workgroup id as well
                    // bda::__ref<scalar_t> scratchPrev = (scratch + prevID).deref();
                    // if (lastInvocation)
                    //     spirv::atomicUMax(scratchPrev.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsReleaseMask, prevReduction|constants_t::LOCAL_COUNT);

                    prefix = binop(prevReduction, prefix);
                }
            }
        }

        scalar_t globalReduction = binop(prefix,localReduction);
        if (lastInvocation)
        {
            bda::__ref<scalar_t> scratchId = (scratch + glsl::gl_WorkGroupID().x).deref();
            spirv::atomicUMax(scratchId.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsReleaseMask, globalReduction|constants_t::GLOBAL_COUNT);
        }

        // get last item from scratch
        const uint32_t lastWorkgroup = glsl::gl_NumWorkGroups().x - 1;
        bda::__ref<scalar_t> scratchLast = (scratch + lastWorkgroup).deref();
        scalar_t value = constants_t::NOT_READY;
        if (lastInvocation)
        {
            // wait until last workgroup does reduction
            while (!(value & constants_t::GLOBAL_COUNT))
            {
                // value = spirv::atomicLoad(scratchLast.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask);
                value = spirv::atomicIAdd(scratchLast.__get_spv_ptr(), spv::ScopeWorkgroup, spv::MemorySemanticsAcquireMask, 0u);
            }
        }
        value = workgroup::Broadcast(value, sharedMemScratchAccessor, Config::WorkgroupSize-1);
        return value & (~constants_t::STATUS_MASK);
    }
};

}

}
}
}

#endif
