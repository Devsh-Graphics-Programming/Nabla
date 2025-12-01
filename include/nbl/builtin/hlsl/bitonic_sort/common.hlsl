#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/concepts/impl/base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bitonic_sort
{

template<typename KeyType, typename ValueType, uint32_t SubgroupSizelog2,typename Comparator>
struct bitonic_sort_config
{
    using key_t = KeyType;
    using value_t = ValueType;
    using comparator_t = Comparator;
    static const uint32_t SubgroupSize = SubgroupSizelog2;                        
};

template<typename Config, class device_capabilities = void>
struct bitonic_sort;


template<typename K, typename V>
inline K get_key(NBL_CONST_REF_ARG(pair<K, V>) kv)
{
    return kv.first;
}

template<typename K, typename V>
inline V get_value(NBL_CONST_REF_ARG(pair<K, V>) kv)
{
    return kv.second;
}

template<typename KeyType>
struct WorkgroupType
{
    using key_t = KeyType;
    using index_t = uint32_t;

    key_t key;
    index_t workgroupRelativeIndex;
};


template<typename KeyType, uint32_t KeyBits, typename StorageType = uint32_t>
struct SubgroupType
{
    using key_t = KeyType;
    using storage_t = StorageType;

    static const uint32_t IndexBits = sizeof(storage_t) * 8u - KeyBits;
    static const storage_t KeyMask = (storage_t(1u) << KeyBits) - 1u;
    static const storage_t IndexMask = ~KeyMask;

    storage_t packed;

    static inline SubgroupType create(key_t key, uint32_t subgroupIndex)
    {
        SubgroupType st;
        st.packed = (storage_t(key) & KeyMask) | (storage_t(subgroupIndex) << KeyBits);
        return st;
    }

    inline key_t getKey() { return key_t(packed & KeyMask); }
    inline uint32_t getSubgroupIndex() { return packed >> KeyBits; }

    inline WorkgroupType<key_t> toWorkgroupType(uint32_t subgroupID, uint32_t elementsPerSubgroup)
    {
        WorkgroupType<key_t> wg;
        wg.key = getKey();
        wg.workgroupRelativeIndex = workgroup::SubgroupContiguousIndex(); 
        return wg;
    }
};

template<typename KeyType>
using SubgroupType27 = SubgroupType<KeyType, 27u>;

template<typename K>
inline K get_key(NBL_CONST_REF_ARG(WorkgroupType<K>) wt)
{
    return wt.key;
}


template<typename K, uint32_t KeyBits, typename StorageType>
inline K get_key(NBL_CONST_REF_ARG(SubgroupType<K, KeyBits, StorageType>) st)
{
    return st.getKey();
}


//template<typename KeyType, typename Comp>
//inline void compareSwap(
//    bool ascending,
//    NBL_REF_ARG(WorkgroupType<KeyType>) a,
//    NBL_REF_ARG(WorkgroupType<KeyType>) b,
//    NBL_CONST_REF_ARG(Comp) comp)
//{
//    const bool swap = comp(b.key, a.key) == ascending;
//    WorkgroupType<KeyType> tmp = a;
//    a = swap ? b : a;
//    b = swap ? tmp : b;
//}


//template<typename KeyType, uint32_t KeyBits, typename StorageType, typename Comp>
//inline void compareSwap(
//    bool ascending,
//    NBL_REF_ARG(SubgroupType<KeyType, KeyBits, StorageType>) a,
//    NBL_REF_ARG(SubgroupType<KeyType, KeyBits, StorageType>) b,
//    NBL_CONST_REF_ARG(Comp) comp)
//{
//    const bool swap = comp(b.getKey(), a.getKey()) == ascending;
//    SubgroupType<KeyType, KeyBits, StorageType> tmp = a;
//    a = swap ? b : a;
//    b = swap ? tmp : b;
//}

template<typename KeyValue, uint32_t Log2N, typename Comparator>
struct LocalPasses
{
      static const uint32_t N = 1u << Log2N;
      void operator()(bool ascending, KeyValue data[N], NBL_CONST_REF_ARG(Comparator) comp);
};

template<typename KeyValue, typename Comparator>
struct LocalPasses<KeyValue, 1, Comparator>
{
      void operator()(bool ascending, KeyValue data[2], NBL_CONST_REF_ARG(Comparator) comp)
      {
            const bool swap = comp(get_key(data[1]), get_key(data[0])) == ascending;

            KeyValue temp = data[0];
            data[0] = swap ? data[1] : data[0];
            data[1] = swap ? temp : data[1];
      }
};

// Specialization for WorkgroupType with 2 elements
template<typename KeyType, typename Comparator>
struct LocalPasses<WorkgroupType<KeyType>, 1, Comparator>
{
    static const uint32_t N = 2;

    void operator()(bool ascending,
        WorkgroupType<KeyType> data[N],
        NBL_CONST_REF_ARG(Comparator) comp)
    {
        const bool swap = comp(get_key(data[1]), get_key(data[0])) == ascending;
        WorkgroupType<KeyType> tmp = data[0];
        data[0] = swap ? data[1] : data[0];
        data[1] = swap ? tmp : data[1];
    }
};

// Specialization for SubgroupType with 2 elements
template<typename KeyType, uint32_t KeyBits, typename StorageType, typename Comparator>
struct LocalPasses<SubgroupType<KeyType, KeyBits, StorageType>, 1, Comparator>
{
    static const uint32_t N = 2;

    void operator()(bool ascending,
        SubgroupType<KeyType, KeyBits, StorageType> data[N],
        NBL_CONST_REF_ARG(Comparator) comp)
    {
        const bool swap = comp(get_key(data[1]), get_key(data[0])) == ascending;
        SubgroupType<KeyType, KeyBits, StorageType> tmp = data[0];
        data[0] = swap ? data[1] : data[0];
        data[1] = swap ? tmp : data[1];
    }
};


} // namespace bitonic_sort
} // namespace hlsl
} // namespace nbl

#endif
