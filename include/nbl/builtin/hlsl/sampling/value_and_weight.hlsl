// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_VALUE_AND_WEIGHT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_VALUE_AND_WEIGHT_INCLUDED_

#include "nbl/builtin/hlsl/sampling/value_and_pdf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename V, typename W>
struct value_and_rcpWeight
{
	using this_t = value_and_rcpWeight<V, W>;
	using scalar_v = typename vector_traits<V>::scalar_type;

	static this_t create(const V _value, const W _rcpWeight)
	{
		this_t retval;
		retval._value = _value;
		retval._rcpWeight = _rcpWeight;
		return retval;
	}

	static this_t create(const scalar_v _value, const W _rcpWeight)
	{
		this_t retval;
		retval._value = hlsl::promote<V>(_value);
		retval._rcpWeight = _rcpWeight;
		return retval;
	}

	V value() { return _value; }
	W rcpWeight() { return _rcpWeight; }

	V _value;
	W _rcpWeight;
};

template<typename V, typename W>
struct value_and_weight
{
	using this_t = value_and_weight<V, W>;
	using scalar_v = typename vector_traits<V>::scalar_type;

	static this_t create(const V _value, const W _weight)
	{
		this_t retval;
		retval._value = _value;
		retval._weight = _weight;
		return retval;
	}

	static this_t create(const scalar_v _value, const W _weight)
	{
		this_t retval;
		retval._value = hlsl::promote<V>(_value);
		retval._weight = _weight;
		return retval;
	}

	V value() { return _value; }
	W weight() { return _weight; }

	V _value;
	W _weight;
};

template<typename V, typename P>
using codomain_and_rcpWeight = value_and_rcpWeight<V, P>;

template<typename V, typename P>
using codomain_and_weight = value_and_weight<V, P>;

template<typename V, typename P>
using domain_and_rcpWeight = value_and_rcpWeight<V, P>;

template<typename V, typename P>
using domain_and_weight = value_and_weight<V, P>;

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
