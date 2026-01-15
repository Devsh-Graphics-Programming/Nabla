#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bitonic_sort
{

template<typename KeyType, typename ValueType, uint32_t SubgroupSizelog2, typename Comparator>
struct bitonic_sort_config
{
    using key_t = KeyType;
    using value_t = ValueType;
    using comparator_t = Comparator;
    static const uint32_t SubgroupSizeLog2 = SubgroupSizelog2;
    static const uint32_t SubgroupSize = 1u << SubgroupSizeLog2;
};

template<typename Config, class device_capabilities = void>
struct bitonic_sort;


template<typename sortable_t, uint32_t Log2N, typename Comparator>
struct LocalPasses
{
    static const uint32_t N = 1u << Log2N;
    void operator()(bool ascending, sortable_t data[N], NBL_CONST_REF_ARG(Comparator) comp);
};

// Specialization for 2 elements (Log2N=1)
template<typename sortable_t, typename Comparator>
struct LocalPasses<sortable_t, 1, Comparator>
{
    static const uint32_t N = 2;

    void operator()(bool ascending, sortable_t data[N], NBL_CONST_REF_ARG(Comparator) comp)
    {
        // For ascending: swap if data[1] < data[0] (put smaller first)
        // For descending: swap if data[0] < data[1] (put larger first)
        const bool needSwap = ascending ? comp(data[1], data[0]) : comp(data[0], data[1]);

        if (needSwap)
        {
            sortable_t temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
    }
};


} // namespace bitonic_sort
} // namespace hlsl
} // namespace nbl

#endif
