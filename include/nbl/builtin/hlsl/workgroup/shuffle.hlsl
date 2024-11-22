#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHUFFLE_INCLUDED_

#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

// TODO: Add other shuffles

// We assume the accessor in the adaptor is clean and unaliased when calling this function, but we don't enforce this after the shuffle

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

// ------------------------------------- Skeletons for implementing other Shuffles --------------------------------

template<typename SharedMemoryAdaptor, typename T>
struct Shuffle
{
    static void __call(NBL_REF_ARG(T) value, uint32_t storeIdx, uint32_t loadIdx, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        // TODO: optimization (optional) where we shuffle in the shared memory available (using rounds)
        sharedmemAdaptor.template set<T>(storeIdx, value);

        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();

        sharedmemAdaptor.template get<T>(loadIdx, value);
    }

    // By default store to threadID in the workgroup
    static void __call(NBL_REF_ARG(T) value, uint32_t loadIdx, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        __call(value, uint32_t(SubgroupContiguousIndex()), loadIdx, sharedmemAdaptor);
    }
};

template<class UnOp, typename SharedMemoryAdaptor, typename T>
struct ShuffleUnOp
{
    static void __call(NBL_REF_ARG(T) value, uint32_t a, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        UnOp unop;
        // TODO: optimization (optional) where we shuffle in the shared memory available (using rounds)
        sharedmemAdaptor.template set<T>(a, value);

        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();

        sharedmemAdaptor.template get<T>(unop(a), value);
    }

    // By default store to threadID's index and load from unop(threadID) 
    static void __call(NBL_REF_ARG(T) value, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        __call(value, uint32_t(SubgroupContiguousIndex()), sharedmemAdaptor);
    }
};

template<class BinOp, typename SharedMemoryAdaptor, typename T>
struct ShuffleBinOp
{
    static void __call(NBL_REF_ARG(T) value, uint32_t a, uint32_t b, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        BinOp binop;
        // TODO: optimization (optional) where we shuffle in the shared memory available (using rounds)
        sharedmemAdaptor.template set<T>(a, value);

        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();

        sharedmemAdaptor.template get<T>(binop(a, b), value);
    }

    // By default first argument of binary op is the thread's ID in the workgroup
    static void __call(NBL_REF_ARG(T) value, uint32_t b, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        __call(value, uint32_t(SubgroupContiguousIndex()), b, sharedmemAdaptor);
    }
};

// ------------------------------------------ ShuffleXor ---------------------------------------------------------------

template<typename SharedMemoryAdaptor, typename T>
void shuffleXor(NBL_REF_ARG(T) value, uint32_t threadID, uint32_t mask, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
{
    return ShuffleBinOp<bit_xor<uint32_t>, SharedMemoryAdaptor, T>::__call(value, threadID, mask, sharedmemAdaptor);
}

template<typename SharedMemoryAdaptor, typename T>
void shuffleXor(NBL_REF_ARG(T) value, uint32_t mask, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
{
    return ShuffleBinOp<bit_xor<uint32_t>, SharedMemoryAdaptor, T>::__call(value, mask, sharedmemAdaptor);
}

}
}
}




#endif