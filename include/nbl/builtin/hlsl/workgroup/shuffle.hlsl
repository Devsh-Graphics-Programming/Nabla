#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHUFFLE_INCLUDED_

#include "nbl/builtin/hlsl/memory_accessor.hlsl"

// TODO: Add other shuffles

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
    static void __call(NBL_REF_ARG(T) value, uint32_t mask, uint32_t threadID, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        sharedmemAccessor.set(threadID, value);
        
        // Wait until all writes are done before reading
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
    
        sharedmemAccessor.get(threadID ^ mask, value);
    }

    static void __call(NBL_REF_ARG(T) value, uint32_t mask, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        __call(value, mask, uint32_t(SubgroupContiguousIndex()), sharedmemAccessor);
    }
};

}
}
}




#endif