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

template<int32_t Dims, typename U=uint16_t, typename T=unsigned_integer_of_size<sizeof(U)*2>::type>
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

// highly specialized function, requires you know the prefices already and that dimension is higher than 1
// TODO: make an even better one that takes precomputed reciprocals and stuff for fast integer division and modulo
template<int32_t Dims, typename U=uint32_t, typename T=conditional<sizeof(U)==2,unsigned_integer_of_size_t<sizeof(U)/2>,uint16_t>::type> // TODO: NBL_REQUIRE Dims>=2
vector<T,Dims> snakeCurveInverse(const U linearIndex, const vector<U,Dims-1> gridDimPrefixProduct)
{
	vector<T,Dims> coord;
	coord[Dims-1] = linearIndex/gridDimPrefixProduct[Dims-2];
	{
		U prevRemainder = linearIndex;
		for (int32_t i=Dims-2; i>0; i--)
		{
			prevRemainder -= gridDimPrefixProduct[i]*coord[i+1];
			coord[i] = prevRemainder/gridDimPrefixProduct[i-1];
		}
		coord[0] = prevRemainder-gridDimPrefixProduct[0]*coord[1];
	}
	coord[Dims-2] = linearIndex-coord[Dims-1]*gridDimPrefixProduct[Dims-2];

	return coord;
}

namespace impl
{
template<int32_t Dims, typename U, typename T>
struct snakeCurveInverse
{
	static vector<T,Dims> __call(const U linearIndex, const vector<T,Dims> gridDim)
	{
		vector<U,Dims-1> gridDimPrefixProduct;
		gridDimPrefixProduct[0] = gridDim[0];
		for (int32_t i=1; i<Dims-1; i++)
			gridDimPrefixProduct[i] = gridDimPrefixProduct[i-1]*gridDim[i];
		return snakeCurveInverse<Dims,U,T>(linearIndex,gridDimPrefixProduct);
	}
};
template<typename U, typename T>
struct snakeCurveInverse<1,U,T>
{
	static vector<T,Dims> __call(const U linearIndex, const vector<T,Dims> gridDim)
	{
		return vector<T,Dims>(linearIndex);
	}
};
}

template<int32_t Dims, typename U=uint32_t, typename T=conditional<sizeof(U)==2,unsigned_integer_of_size_t<sizeof(U)/2>,uint16_t>::type>
vector<T,Dims> snakeCurveInverse(const U linearIndex, const vector<T,Dims> gridDim)
{
	return impl::snakeCurveInverse<Dims,U,T>::__call(linearIndex,gridDim);
}

}
}
}
#endif