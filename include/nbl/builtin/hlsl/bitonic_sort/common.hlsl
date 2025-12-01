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


template<typename KeyValue, uint32_t Log2N, typename Comparator>
struct LocalPasses
{
    static const uint32_t N = 1u << Log2N;
    void operator()(bool ascending, KeyValue data[N], NBL_CONST_REF_ARG(Comparator) comp);
};

template<typename KeyValue, typename Comparator>
struct LocalPasses<KeyValue, 1, Comparator>
{
    static const uint32_t N = 2;

    void operator()(bool ascending, KeyValue data[N], NBL_CONST_REF_ARG(Comparator) comp)
    {
        const bool shouldSwap = comp(data[1], data[0]) == ascending;

        KeyValue temp = data[0];
        data[0] = shouldSwap ? data[1] : data[0];
        data[1] = shouldSwap ? temp : data[1];
    }
};


} // namespace bitonic_sort
} // namespace hlsl
} // namespace nbl

#endif
