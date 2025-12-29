// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_OBB_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_OBB_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace shapes
{

template<int16_t D=3, typename Scalar=float32_t>
struct OBB
{
	using scalar_t = Scalar;
	using point_t = vector<Scalar,D>;

	static OBB createAxisAligned(point_t mid, point_t len)
	{
		OBB ret;
		ret.mid = mid;
		ret.ext = len * 0.5f;
		for (auto dim_i = 0; dim_i < D; dim_i++)
		{
			ret.axes[dim_i] = point_t(0);
			ret.axes[dim_i][dim_i] = 1;
		}
		return ret;
	}

	point_t mid;
	std::array<point_t, D> axes;
	point_t ext;
};

}
}
}

#endif
