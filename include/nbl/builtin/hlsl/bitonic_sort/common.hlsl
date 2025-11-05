#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/math/intutil.hlsl>
#include <nbl/builtin/hlsl/utility.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bitonic_sort
{

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
	if (takePartnerLo)
		loPair = partnerLoPair;

	const bool hiSelfSmaller = comp(hiPair.first, partnerHiPair.first);
	const bool takePartnerHi = takeLarger ? hiSelfSmaller : !hiSelfSmaller;
	if (takePartnerHi)
		hiPair = partnerHiPair;
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

	if (doSwap)
		swap(loPair, hiPair);
}
}
}
}

#endif
