#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BITONIC_SORT_INCLUDED_

#include "nbl/builtin/hlsl/bitonic_sort/common.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"

namespace nbl
{
    namespace hlsl
    {
		namespace bitonic_sort
        {

            // -----------------------------------------------------------------------------------------------------------------------------------------------------------------
            template<bool Inverse, typename Scalar, class device_capabilities = void>
			struct bitonic_sort
            {
                static void __call(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi);
            };

        }
    }
}

#endif