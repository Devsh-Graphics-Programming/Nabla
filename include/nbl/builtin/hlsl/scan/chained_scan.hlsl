#ifndef _NBL_HLSL_SCAN_CHAINED_SCAN_INCLUDED_
#define _NBL_HLSL_SCAN_CHAINED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

groupshared bool sIsLocked;

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

    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = 1u << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t PreloadedDataCount = uint16_t(VirtualWorkgroupSize / WorkgroupSize);

    static WorkgroupDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> create(const uint64_t inputBuf, const uint64_t outputBuf, const uint16_t workgroupId)
    {
        WorkgroupDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> retval;
        const uint32_t workgroupOffset = workgroupId * VirtualWorkgroupSize * sizeof(dtype_t);
        retval.accessor = DoubleLegacyBdaAccessor<dtype_t>::create(inputBuf/* + workgroupOffset*/, outputBuf /*+ workgroupOffset*/);
        retval.workgroupID = workgroupId;
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
            accessor.get((workgroupID + idx) * WorkgroupSize + invocIx, preloaded[idx]);
    }
    void unload()
    {
        const uint16_t invocIx = workgroup::SubgroupContiguousIndex();
        NBL_UNROLL
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            accessor.set((workgroupID + idx) * WorkgroupSize + invocIx, preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DoubleLegacyBdaAccessor<dtype_t> accessor;
    dtype_t preloaded[PreloadedDataCount];
    uint32_t workgroupID;   // TODO: remove possibly?, current problem is that adding the workgroup offset directly to address doesn't work
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

    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = 1u << Config::WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvoc = uint16_t(Config::VirtualWorkgroupSize / WorkgroupSize);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxSpinCount = uint16_t(4u);

    template<class DataAccessor, class ScratchAccessor, class ReductionAccessor, class WorkgroupCounter>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(ReductionAccessor) workgroupReduction, NBL_REF_ARG(WorkgroupCounter) workgroupCounter)
    {        
        const uint16_t invocIx = workgroup::SubgroupContiguousIndex();
        if (!invocIx)
        {
            const uint32_t id = workgroupCounter.atomicAdd(0u, 1u);
            scratchAccessor.template set<uint32_t, uint32_t>(0u, id);
            sIsLocked = true;
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();
        // spirv::memoryBarrier(spv::ScopeDevice, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);

        uint16_t workgroupId;
        scratchAccessor.template get<uint32_t, uint32_t>(0u, workgroupId);
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        binop_t binop;
        using wg_data_proxy_t = WorkgroupDataProxy<Config::WorkgroupSizeLog2,Config::VirtualWorkgroupSize,Config::ItemsPerInvocation_0>;
        wg_data_proxy_t wgDataAccessor = wg_data_proxy_t::create(dataAccessor.getInputBufAddr(), dataAccessor.getOutputBufAddr(), workgroupId);
        scalar_t currGroupReduction;
        {
            wgDataAccessor.preload();

            // TODO: double check this but it should be the last element of the last workgroup thread
            // don't know what it's like if virtual workgroup size doesn't divide by workgroup size exactly
            const uint32_t lastInvocIx = glsl::gl_SubgroupSize() * glsl::gl_NumSubgroups() - 1u;
            scalar_t lastElem;
            if (invocIx == lastInvocIx)
                lastElem = wgDataAccessor.preloaded[wg_data_proxy_t::PreloadedDataCount-1u][Config::ItemsPerInvocation_0-1u];

            if (Exclusive)
                workgroup2::exclusive_scan<Config,BinOp,device_capabilities>::template __call<wg_data_proxy_t, ScratchAccessor>(wgDataAccessor, scratchAccessor);
            else
                workgroup2::inclusive_scan<Config,BinOp,device_capabilities>::template __call<wg_data_proxy_t, ScratchAccessor>(wgDataAccessor, scratchAccessor);
            scratchAccessor.workgroupExecutionAndMemoryBarrier();

            // wgDataAccessor.unload();    // TODO: maybe we don't have to unload, just write once after everything is done

            currGroupReduction = wgDataAccessor.preloaded[wg_data_proxy_t::PreloadedDataCount-1u][Config::ItemsPerInvocation_0-1u];
            if (Exclusive)
                currGroupReduction = binop(currGroupReduction, lastElem);
            if (invocIx == lastInvocIx)
                scratchAccessor.template set<uint32_t, uint32_t>(0u, currGroupReduction);
            scratchAccessor.workgroupExecutionAndMemoryBarrier();

            scratchAccessor.template get<uint32_t, uint32_t>(0u, currGroupReduction);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        if (!invocIx)
        {
            const scalar_t storeVal = hlsl::mix(Flag_Inclusive, Flag_Reduction, workgroupId > 0u) | currGroupReduction << Flag_Shift;
            workgroupReduction.atomicExchange(workgroupId, storeVal);
        }

        if (workgroupId > 0)
        {
            bool locked = sIsLocked;
            scratchAccessor.workgroupExecutionAndMemoryBarrier();

            scalar_t prevReduction = 0u;
            uint16_t lookbackIx = workgroupId - uint16_t(1u);

            while (locked)
            {
                // lookback: try to get reduction from previous workgroups
                if (!invocIx)
                {
                    uint16_t spinCount = uint16_t(0u);
                    [loop]
                    while (spinCount < MaxSpinCount)
                    {
                        scalar_t flagPayload;
                        workgroupReduction.get(lookbackIx, flagPayload);

                        if ((flagPayload & Flag_Mask) > Flag_NotReady)
                        {
                            spinCount = uint16_t(0u);
                            prevReduction = binop(prevReduction, flagPayload >> Flag_Shift);
                            if ((flagPayload & Flag_Mask) == Flag_Inclusive)
                            {
                                const scalar_t storeVal = Flag_Inclusive | (binop(prevReduction, currGroupReduction) << Flag_Shift);
                                workgroupReduction.atomicExchange(workgroupId, storeVal);
                                scratchAccessor.template set<uint32_t, uint32_t>(0u, prevReduction);
                                sIsLocked = false;
                                break;
                            }
                            else
                                lookbackIx--;
                        }
                        else
                            spinCount++;
                    }

                    // broadcast id and prepare to do reduction ourselves
                    if (spinCount == MaxSpinCount)
                        scratchAccessor.template set<uint32_t, uint32_t>(1u, lookbackIx);
                }
                scratchAccessor.workgroupExecutionAndMemoryBarrier();

                locked = sIsLocked;
                scratchAccessor.workgroupExecutionAndMemoryBarrier();
                if (locked)
                {
                    // do reduction for lookbackIx workgroup
                    uint16_t fallbackGroupId;
                    scratchAccessor.template get<uint32_t, uint32_t>(1u, fallbackGroupId);

                    wg_data_proxy_t fallbackDataAccessor = wg_data_proxy_t::create(dataAccessor.getInputBufAddr(), dataAccessor.getOutputBufAddr(), fallbackGroupId);
                    fallbackDataAccessor.preload();
                    scalar_t fallbackReduction = workgroup2::reduction<Config,BinOp,device_capabilities>::template __call<wg_data_proxy_t, ScratchAccessor>(fallbackDataAccessor, scratchAccessor);
                    scratchAccessor.workgroupExecutionAndMemoryBarrier();

                    if (!invocIx)
                    {
                        const scalar_t storeVal = hlsl::mix(Flag_Inclusive, Flag_Reduction, fallbackGroupId > 0u) | (fallbackReduction << Flag_Shift);
                        const scalar_t fallbackPayload = workgroupReduction.atomicMax(fallbackGroupId, storeVal);

                        prevReduction = binop(prevReduction, hlsl::mix(fallbackReduction, fallbackPayload >> Flag_Shift, fallbackPayload > scalar_t(0.0)));
                        if (!fallbackGroupId || (fallbackPayload & Flag_Mask) == Flag_Inclusive)
                        {
                            const scalar_t storeVal = Flag_Inclusive | (binop(prevReduction, currGroupReduction) << Flag_Shift);
                            workgroupReduction.atomicExchange(workgroupId, storeVal);
                            scratchAccessor.template set<uint32_t, uint32_t>(0u, prevReduction);
                            sIsLocked = false;
                        }
                        else
                            lookbackIx--;
                    }
                    scratchAccessor.workgroupExecutionAndMemoryBarrier();

                    locked = sIsLocked;
                    scratchAccessor.workgroupExecutionAndMemoryBarrier();
                }
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        dataAccessor.initAtWorkgroupID(workgroupId);

        scalar_t prevReduction = 0u;
        if (workgroupId > 0)
            scratchAccessor.template get<uint32_t, uint32_t>(0u, prevReduction);

        NBL_UNROLL
        for (uint16_t idx = 0; idx < wg_data_proxy_t::PreloadedDataCount; idx++)
        {
            vector_t data = wgDataAccessor.preloaded[idx];
            NBL_UNROLL
            for (uint16_t i = 0; i < Config::ItemsPerInvocation_0; i++)
                data[i] = binop(prevReduction, data[i]);
            dataAccessor.template set<vector_t, uint32_t>(idx * WorkgroupSize + invocIx, data);
        }
    }
};
}

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(workgroup2::is_configuration_v<Config>)
struct inclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    // TODO: might want new concept for ReductionAccessor
    template<class DataAccessor, class ScratchAccessor, class ReductionAccessor, class WorkgroupCounter NBL_FUNC_REQUIRES(workgroup2::ArithmeticDataAccessor<DataAccessor,scalar_t> && workgroup2::ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(ReductionAccessor) workgroupReduction, NBL_REF_ARG(WorkgroupCounter) counter)
    {
        impl::Scan<Config,BinOp,false,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor, workgroupReduction, counter);
    }
};

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(workgroup2::is_configuration_v<Config>)
struct exclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    template<class DataAccessor, class ScratchAccessor, class ReductionAccessor, class WorkgroupCounter NBL_FUNC_REQUIRES(workgroup2::ArithmeticDataAccessor<DataAccessor,scalar_t> && workgroup2::ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor, NBL_REF_ARG(ReductionAccessor) workgroupReduction, NBL_REF_ARG(WorkgroupCounter) counter)
    {
        impl::Scan<Config,BinOp,true,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor, workgroupReduction, counter);
    }
};

}
}
}

#endif
