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

	NBL_CONSTEXPR_STATIC_INLINE OBB create(point_t mid, point_t ext, const point_t axes[D])
	{
		point_t obbScale = ext * 2.0f;
		point_t axesScale[D];
		for (int dim_i = 0; dim_i < D; dim_i++)
		{
			axesScale[dim_i] = axes[dim_i] * obbScale[dim_i];
		}
		OBB ret;
		for (int row_i = 0; row_i < D; row_i++)
		{
		  for (int col_i = 0; col_i < D; col_i++)
		  {
				ret.transform[row_i][col_i] = axesScale[col_i][row_i];
		  }
		}
		for (int dim_i = 0; dim_i < D; dim_i++)
		{
			scalar_t sum = 0; 
			for (int dim_j = 0; dim_j < D; dim_j++)
			{
				sum += axesScale[dim_j][dim_i];
			}
			ret.transform[dim_i][D] = mid[dim_i] - (0.5 * sum);
		}
		return ret;
	  
	}

	NBL_CONSTEXPR_STATIC_INLINE OBB createAxisAligned(point_t mid, point_t len)
	{
		point_t axes[D];
		for (auto dim_i = 0; dim_i < D; dim_i++)
		{
			axes[dim_i] = point_t(0);
			axes[dim_i][dim_i] = 1;
		}
		return create(mid, len * 0.5f, axes);
	}

	matrix<scalar_t, D, D + 1> transform;
};

}
}
}

#endif
