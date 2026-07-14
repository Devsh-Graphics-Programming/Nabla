#ifndef _NBL_HLSL_SCAN_CHAINED_SCAN_INCLUDED_
#define _NBL_HLSL_SCAN_CHAINED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/workgroup2/shared_scan.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{

namespace impl
{
template<class Config, class BinOp, bool Exclusive, class device_capabilities>
struct scan
{
    using scalar_t = typename BinOp::type_t;
    using vector_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type

    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_NotReady = 0;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_Reduction = 1;    // workgroup only has local reduction ready
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_Inclusive = 2;    // workgroup has summed all preceding groups and added to own sum
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_Mask = 3;
    
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Flag_Shift = 2;

    // TODO: I think we need separate DataAccessor classes between device-wide and workgroup scan
    
    // TODO: double check op, it's all plus right now

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(DataAccessor) workgroupReduction)   // TODO: need different type for workgroupReduction
    {
        // TODO: what to do with config?
        
        const uint16_t invocIx = workgroup::SubgroupContiguousIndex();
        const uint16_t workgroupId = glsl::gl_WorkGroupID();

        // TODO: workgroup scan here, don't want to deal with that now
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        const scalar_t currGroupReduction = // TODO get last elem of curr workgroup;
        if (!invocIx)
        {
            const scalar_t storeVal = hlsl::mix(Flag_Inclusive, Flag_Reduction, workgroupId > 0u) | currGroupReduction << Flag_Shift;
            workgroupReduction.atomicExchange(workgroupId, storeVal);
        }

        // lookback
        if (workgroupId && !invocIx)
        {
            scalar_t prevReduction = 0u;
            uint16_t lookbackIx = workgroupId - uint16_t(1u);

            while (true)    // TODO: check if lookbackIx < 0?
            {
                scalar_t flagPayload;
                workgroupReduction.get(lookbackIx, flagPayload);

                if ((flagPayload & Flag_Mask) > Flag_NotReady)
                {
                    prevReduction += flagPayload >> Flag_Shift;
                    if ((flagPayload & Flag_Mask) == Flag_Inclusive)
                    {
                        const scalar_t groupReduction = // TODO get last elem of workgroup index lookbackIx;
                        const scalar_t storeVal = Flag_Inclusive | ((prevReduction + groupReduction) << Flag_Shift);
                        workgroupReduction.atomicExchange(workgroupId, storeVal);
                        scratchAccessor.set(0u, prevReduction);
                        break;
                    }
                    else
                        lookbackIx--;
                }
                else
                    lookbackIx--;
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        scalar_t prevReduction;
        scratchAccessor.get(0u, prevReduction);
        prevReduction += currGroupReduction;

        // TODO: propagate correctly when data accessor done
        dataAccessor[invocIx] += prevReduction;
    }
};
}

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct inclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        impl::scan<Config,BinOp,false,Config::LevelCount,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor);
    }
};

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct exclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
    }
};

}
}
}

#endif
