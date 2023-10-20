// TODO: Delete this file!
// This file is temporary file that defines all of the dependencies on PR #519
// and should be deleted as soon as that's merged.
#ifndef _NBL_BUILTIN_HLSL_BLIT_TEMP_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_TEMP_INCLUDED_


namespace nbl
{
namespace hlsl
{

namespace spirv
{
[[vk::ext_instruction(/* OpUMulExtended */ 151)]]
uint32_t2 umulExtended(uint32_t v0, uint32_t v1);
}


namespace binops
{
template<typename T>
struct add
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs + rhs;
    }

    static T identity()
    {
        return 0;
    }
};
}


namespace workgroup
{
    // This is slow naive scan but it doesn't matter as this file is going to
    // be nuked. The interface is different than the one suggested in PR #519
    // because right now there's no easy hack-free way to access
    // gl_localInvocationID globally.
    template<uint32_t WorkGroupSize, typename T, class Binop, class SharedAccessor>
    T inclusive_scan(T value, NBL_REF_ARG(SharedAccessor) sharedAccessor, uint32_t localInvocationID)
    {
        for (uint32_t i = 0; i < firstbithigh(WorkGroupSize); ++i)
        {
            sharedAccessor.main.set(localInvocationID, value);
            GroupMemoryBarrierWithGroupSync();
            if (localInvocationID >= (1 << i))
            {
                value = Binop(sharedAccessor.get(localInvocationID - (1 << i)), value);
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}


}
}

#endif