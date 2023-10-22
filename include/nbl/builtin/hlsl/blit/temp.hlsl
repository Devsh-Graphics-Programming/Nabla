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
// DXC fails with uint32_t2 result, so we have to use a struct.
struct umul_result_t { uint32_t lsb; uint32_t msb; };
[[vk::ext_instruction(/* OpUMulExtended */ 151)]]
umul_result_t umulExtended(uint32_t v0, uint32_t v1);
}

namespace workgroup
{
    // This is slow naive scan but it doesn't matter as this file is going to
    // be nuked. The interface is different than the one suggested in PR #519
    // because right now there's no easy hack-free way to access
    // gl_localInvocationID globally.
    template<uint32_t WorkGroupSize, typename T, typename Binop, typename SharedAccessor>
    T inclusive_scan(T value, NBL_REF_ARG(SharedAccessor) sharedAccessor, uint32_t localInvocationIndex)
    {
        Binop binop;
        for (uint32_t i = 0; i < firstbithigh(WorkGroupSize); ++i)
        {
            sharedAccessor.main.set(localInvocationIndex, value);
            GroupMemoryBarrierWithGroupSync();
            if (localInvocationIndex >= (1u << i))
            {
                value = binop(sharedAccessor.main.get(localInvocationIndex - (1u << i)), value);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        return value;
    }
}


}
}

#endif