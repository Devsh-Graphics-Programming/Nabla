// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_NBL_HLSL_MULTI_DIMENSIONAL_ARRAY_ADDRESSING_INCLUDED_
#define _NBL_NBL_HLSL_MULTI_DIMENSIONAL_ARRAY_ADDRESSING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ndarray_addressing
{

template<int32_t Dims, typename U=uint16_t, typename T=typename unsigned_integer_of_size<sizeof(U)*2>::type>
T snakeCurve(const vector<U,Dims> coordinate, const vector<U,Dims-1> extents)
{
	T retval = _static_cast<T>(coordinate[Dims-1]);
	for (int32_t i=Dims-2; i>=0; i--)
	{
		retval *= _static_cast<T>(extents[i]);
		retval += _static_cast<T>(coordinate[i]);
	}
	return retval;
}

// TODO: make an even better one that takes precomputed reciprocals and stuff for fast integer division and modulo
// https://github.com/milakov/int_fastdiv
template<int32_t Dims, typename U=uint32_t, typename T=typename conditional<sizeof(U)==2,unsigned_integer_of_size_t<sizeof(U)/2>,uint16_t>::type>
vector<T,Dims> snakeCurveInverse(const U linearIndex, const vector<T,Dims> gridDim)
{
	vector<T,Dims> coord;
	{
		U prev = linearIndex;
		U next;
		for (int32_t i=0; i<Dims-1; i++)
		{
			next = prev/gridDim[i];
			coord[i] = prev-next*gridDim[i];
			prev = next;
		}
		coord[Dims-1] = prev;
	}
	return coord;
}

}
}
}
#endif