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

template<typename SharedMemoryAdaptor, typename T>
struct shuffleXor
{
    static void __call(NBL_REF_ARG(T) value, uint32_t mask, uint32_t threadID, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        sharedmemAdaptor.template set<T>(threadID, value);
        
        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
    
        sharedmemAdaptor.template get<T>(threadID ^ mask, value);
    }

    static void __call(NBL_REF_ARG(T) value, uint32_t mask, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        __call(value, mask, uint32_t(SubgroupContiguousIndex()), sharedmemAdaptor);
    }
};

// Vector specialization
template<typename SharedMemoryAdaptor, typename T, uint32_t N>
struct shuffleXor<SharedMemoryAdaptor, vector <T, N> >
{
    static enable_if_t<N <= 4> __call(NBL_REF_ARG(vector <T, N>) value, uint32_t mask, uint32_t threadID, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        sharedmemAdaptor.template set<T>(threadID, value);
        
        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
    
        sharedmemAdaptor.template get<T>(threadID ^ mask, value);
    }

    static enable_if_t<N <= 4> __call(NBL_REF_ARG(vector <T, N>) value, uint32_t mask, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        __call(value, mask, uint32_t(SubgroupContiguousIndex()), sharedmemAdaptor);
    }
};

}
}
}




#endif