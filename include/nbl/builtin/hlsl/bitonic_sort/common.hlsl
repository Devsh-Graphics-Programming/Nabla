#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bitonic_sort
{

template<typename sortable_t, uint32_t Log2N, typename Comparator>
struct LocalPasses
{
    static const uint32_t N = 1u << Log2N;
    void operator()(bool ascending, sortable_t data[N], NBL_CONST_REF_ARG(Comparator) comp);
};

} // namespace bitonic_sort
} // namespace hlsl
} // namespace nbl

#endif
