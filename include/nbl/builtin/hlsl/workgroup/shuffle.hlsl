#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHUFFLE_INCLUDED_

#include "nbl/builtin/hlsl/memory_accessor.hlsl"

// TODO: Add other shuffles
// TODO: Consider adding an enable_if or static assert that 1 <= N <= 4 and that Scalar is a proper scalar type
// TODO: Consider adding version that doesn't take a precomputed threadID and instead calls workgroup::SubgroupContiguousIndex

// Unlike subgroups we pass a precomputed threadID so we don't go around recomputing it every time
// We assume the accessor in the adaptor is clean and unaliased when calling this function, but we don't enforce this after the shuffle

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{


template<typename SharedMemoryAccessor, typename T>
struct shuffleXor
{
    static void __call(NBL_REF_ARG(T) value, uint32_t mask, uint32_t threadID, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
    {
        sharedmemAdaptor.set(threadID, value);
        
        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
    
        sharedmemAdaptor.get(threadID ^ mask, value);
    }

    static void __call(NBL_REF_ARG(T) value, uint32_t mask, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
    {
        __call(value, mask, uint32_t(SubgroupContiguousIndex()), sharedmemAdaptor);
    }
};

/*

template<typename SharedMemoryAccessor, typename T>
void shuffleXor(NBL_REF_ARG(T) value, uint32_t mask, uint32_t threadID, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
{
    sharedmemAdaptor.set(threadID, value);
        
    // Wait until all writes are done before reading
    sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
    
    sharedmemAdaptor.get(threadID ^ mask, value);
}

*/

}
}
}




#endif