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
template<uint16_t WorkgroupSizeLog2, uint16_t VirtualWorkgroupSize, uint16_t ItemsPerInvocation>
struct WorkgroupDataProxy
{
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t PreloadedDataCount = VirtualWorkgroupSize / WorkgroupSize;

    static WorkgroupDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> create(const uint64_t inputBuf, const uint64_t outputBuf)
    {
        WorkgroupDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> retval;
        const uint32_t workgroupOffset = glsl::gl_WorkGroupID().x * VirtualWorkgroupSize * sizeof(dtype_t);
        retval.accessor = DoubleLegacyBdaAccessor<dtype_t>::create(inputBuf + workgroupOffset, outputBuf + workgroupOffset);
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = preloaded[ix>>WorkgroupSizeLog2];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        preloaded[ix>>WorkgroupSizeLog2] = value;
    }

    void preload()
    {
        const uint16_t invocIx = workgroup::SubgroupContiguousIndex();
        NBL_UNROLL
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            accessor.get(idx * WorkgroupSize + invocIx, preloaded[idx]);
    }
    void unload()
    {
        const uint16_t invocIx = workgroup::SubgroupContiguousIndex();
        NBL_UNROLL
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            accessor.set(idx * WorkgroupSize + invocIx, preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DoubleLegacyBdaAccessor<dtype_t> accessor;
    dtype_t preloaded[PreloadedDataCount];
};

template<class Config, class BinOp, bool Exclusive, class device_capabilities>  // TODO: Config is same as workgroup2 stuff?
struct Scan
{
    using scalar_t = typename BinOp::type_t;
    using vector_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using binop_t = BinOp;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_NotReady = 0;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_Reduction = 1;    // workgroup only has local reduction ready
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_Inclusive = 2;    // workgroup has summed all preceding groups and added to own sum
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Flag_Mask = 3;
    
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Flag_Shift = 2;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1u) << Config::WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvoc = Config::VirtualWorkgroupSize / WorkgroupSize;

    template<class DataAccessor, class ScratchAccessor, class ReductionAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(ReductionAccessor) workgroupReduction)
    {        
        const uint16_t invocIx = workgroup::SubgroupContiguousIndex();
        const uint16_t workgroupId = glsl::gl_WorkGroupID();
        binop_t binop;

        scalar_t currGroupReduction;
        {
            using data_proxy_t = WorkgroupDataProxy<Config::WorkgroupSizeLog2,Config::VirtualWorkgroupSize,Config::ItemsPerInvocation_0>;
            data_proxy_t wgDataAccessor = data_proxy_t::create(dataAccessor.getInputBufAddr(), dataAccessor.getOutputBufAddr());
            wgDataAccessor.preload();

            if (Exclusive)
                workgroup2::exclusive_scan<Config,BinOp,device_capabilities>::template __call<data_proxy_t, ScratchAccessor>(wgDataAccessor, scratchAccessor);
            else
                workgroup2::inclusive_scan<Config,BinOp,device_capabilities>::template __call<data_proxy_t, ScratchAccessor>(wgDataAccessor, scratchAccessor);
            scratchAccessor.workgroupExecutionAndMemoryBarrier();

            wgDataAccessor.unload();    // TODO: maybe we don't have to unload, just write once after everything is done

            // TODO: double check this but it should be the last element of the last workgroup thread
            // don't know what it's like if virtual workgroup size doesn't divide by workgroup size exactly
            if (invocIx == glsl::gl_SubgroupSize() * glsl::gl_NumSubgroups() - 1u)
                currGroupReduction = wgDataAccessor.preloaded[data_proxy_t::PreloadedDataCount-1u][Config::ItemsPerInvocation_0-1u];
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

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
                    prevReduction = binop(prevReduction, flagPayload >> Flag_Shift);
                    if ((flagPayload & Flag_Mask) == Flag_Inclusive)
                    {
                        const scalar_t storeVal = Flag_Inclusive | (binop(prevReduction, currGroupReduction) << Flag_Shift);
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
        prevReduction = binop(prevReduction, currGroupReduction);

        NBL_UNROLL
        for (uint16_t idx = 0; idx < ItemsPerInvoc; idx++)
        {
            dtype_t data;
            dataAccessor.template get<dtype_t, uint16_t>(idx * WorkgroupSize + invocIx, data);
            NBL_UNROLL
            for (uint16_t i = 0; i < Config::ItemsPerInvocation_0; i++)
                data[i] = binop(prevReduction, data[i]);
            dataAccessor.template set<dtype_t, uint16_t>(idx * WorkgroupSize + invocIx, data);
        }
    }
};
}

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct inclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    // TODO: might want new concept for ReductionAccessor
    template<class DataAccessor, class ScratchAccessor, class ReductionAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(ReductionAccessor) workgroupReduction)
    {
        impl::Scan<Config,BinOp,false,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor, workgroupReduction);
    }
};

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct exclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    template<class DataAccessor, class ScratchAccessor, class ReductionAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(ReductionAccessor) workgroupReduction)
    {
        impl::Scan<Config,BinOp,true,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor, workgroupReduction);
    }
};

}
}
}

#endif
