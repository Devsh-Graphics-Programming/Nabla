#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/math/intutil.hlsl>
#include <nbl/builtin/hlsl/pair.hlsl>

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
    const bool loSelfSmaller = comp(loKey, partnerLoKey);
    const bool takePartnerLo = takeLarger ? loSelfSmaller : !loSelfSmaller;
    loKey = takePartnerLo ? partnerLoKey : loKey;
    loVal = takePartnerLo ? partnerLoVal : loVal;

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



template<typename KeyType, typename ValueType, typename Comparator>
void compareExchangeWithPartner(
bool takeLarger,
NBL_REF_ARG(pair<KeyType, ValueType>) loPair,
NBL_CONST_REF_ARG(pair<KeyType, ValueType>) partnerLoPair,
NBL_REF_ARG(pair<KeyType, ValueType>) hiPair,
NBL_CONST_REF_ARG(pair<KeyType, ValueType>) partnerHiPair,
NBL_CONST_REF_ARG(Comparator) comp)
{
    const bool loSelfSmaller = comp(loPair.first, partnerLoPair.first);
    const bool takePartnerLo = takeLarger ? loSelfSmaller : !loSelfSmaller;
    loPair.first = takePartnerLo ? partnerLoPair.first : loPair.first;
    loPair.second = takePartnerLo ? partnerLoPair.second : loPair.second;

    const bool hiSelfSmaller = comp(hiPair.first, partnerHiPair.first);
    const bool takePartnerHi = takeLarger ? hiSelfSmaller : !hiSelfSmaller;
    hiPair.first = takePartnerHi ? partnerHiPair.first : hiPair.first;
    hiPair.second = takePartnerHi ? partnerHiPair.second : hiPair.second;
}

template<typename KeyType, typename ValueType, typename Comparator>
void compareSwap(
bool ascending,
NBL_REF_ARG(pair<KeyType, ValueType>) loPair,
NBL_REF_ARG(pair<KeyType, ValueType>) hiPair,
NBL_CONST_REF_ARG(Comparator) comp)
{
    const bool shouldSwap = comp(hiPair.first, loPair.first);
    const bool doSwap = (shouldSwap == ascending);

    KeyType tempKey = loPair.first;
    ValueType tempVal = loPair.second;
    loPair.first = doSwap ? hiPair.first : loPair.first;
    loPair.second = doSwap ? hiPair.second : loPair.second;
    hiPair.first = doSwap ? tempKey : hiPair.first;
    hiPair.second = doSwap ? tempVal : hiPair.second;
}

template<typename KeyType, typename ValueType>
void swap(
NBL_REF_ARG(pair<KeyType, ValueType>) loPair,
NBL_REF_ARG(pair<KeyType, ValueType>) hiPair)
{
    pair<KeyType, ValueType> temp = loPair;
    loPair = hiPair;
    hiPair = temp;
}
}
}
}

#endif
