#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/math/intutil.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bitonic_sort
{

template<typename KeyType, typename ValueType, typename Comparator>
void compareExchangeWithPartner(
    bool takeLarger,
    NBL_REF_ARG(KeyType) loKey,
    NBL_CONST_REF_ARG(KeyType) partnerLoKey,
    NBL_REF_ARG(KeyType) hiKey,
    NBL_CONST_REF_ARG(KeyType) partnerHiKey,
    NBL_REF_ARG(ValueType) loVal,
    NBL_CONST_REF_ARG(ValueType) partnerLoVal,
    NBL_REF_ARG(ValueType) hiVal,
    NBL_CONST_REF_ARG(ValueType) partnerHiVal,
    NBL_CONST_REF_ARG(Comparator) comp)
{
    // Process lo pair
    const bool loSelfSmaller = comp(loKey, partnerLoKey);
    const bool takePartnerLo = takeLarger ? loSelfSmaller : !loSelfSmaller;
    loKey = takePartnerLo ? partnerLoKey : loKey;
    loVal = takePartnerLo ? partnerLoVal : loVal;

    // Process hi pair
    const bool hiSelfSmaller = comp(hiKey, partnerHiKey);
    const bool takePartnerHi = takeLarger ? hiSelfSmaller : !hiSelfSmaller;
    hiKey = takePartnerHi ? partnerHiKey : hiKey;
    hiVal = takePartnerHi ? partnerHiVal : hiVal;
}


template<typename KeyType, typename ValueType, typename Comparator>
void compareSwap(
    bool ascending,
    NBL_REF_ARG(KeyType) loKey,
    NBL_REF_ARG(KeyType) hiKey,
    NBL_REF_ARG(ValueType) loVal,
    NBL_REF_ARG(ValueType) hiVal,
    NBL_CONST_REF_ARG(Comparator) comp)
{
    const bool shouldSwap = comp(hiKey, loKey);

    const bool doSwap = (shouldSwap == ascending);

    KeyType tempKey = loKey;
    loKey = doSwap ? hiKey : loKey;
    hiKey = doSwap ? tempKey : hiKey;

    ValueType tempVal = loVal;
    loVal = doSwap ? hiVal : loVal;
    hiVal = doSwap ? tempVal : hiVal;
}

template<typename KeyType, typename ValueType>
void swap(
    NBL_REF_ARG(KeyType) loKey,
    NBL_REF_ARG(KeyType) hiKey,
    NBL_REF_ARG(ValueType) loVal,
    NBL_REF_ARG(ValueType) hiVal)
{
    KeyType tempKey = loKey;
    loKey = hiKey;
    hiKey = tempKey;

    ValueType tempVal = loVal;
    loVal = hiVal;
    hiVal = tempVal;
}

}
}
}

#endif
