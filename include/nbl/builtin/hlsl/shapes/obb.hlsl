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
		for (int16_t row_i = 0; row_i < D; row_i++)
		{
		  for (int16_t col_i = 0; col_i < D; col_i++)
		  {
				ret.transform[row_i][col_i] = axesScale[col_i][row_i];
		  }
		}
		for (int16_t dim_i = 0; dim_i < D; dim_i++)
		{
			scalar_t sum = 0; 
			for (int16_t dim_j = 0; dim_j < D; dim_j++)
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
		for (int16_t dim_i = 0; dim_i < D; dim_i++)
		{
			axes[dim_i] = point_t(0);
			axes[dim_i][dim_i] = 1;
		}
		return create(mid, len * 0.5f, axes);
	}

	matrix<scalar_t, D, D + 1> transform;
};

// Decomposed OBB view: caches columns and minCorner for fast vertex queries.
// Same 12-float footprint as float32_t3x4, but laid out for branchless
// corner computation.
// Decomposed OBB view: caches columns and minCorner for fast vertex queries.
// columns[i] are the full OBB edge vectors; minCorner is the world position
// of corner 0b000 = center - 0.5*(col0+col1+col2).
template<typename Scalar=float32_t>
struct OBBView
{
	using scalar_t = Scalar;
	using vec3_t = vector<Scalar, 3>;
	using mat3_t = matrix<Scalar, 3, 3>;

	mat3_t columns;
	vec3_t minCorner;

	static OBBView create(matrix<scalar_t, 3, 4> modelMatrix)
	{
		matrix<scalar_t, 4, 3> m = transpose(modelMatrix);
		OBBView v;
		v.columns = mat3_t(m[0].xyz, m[1].xyz, m[2].xyz);
		v.minCorner = m[3].xyz - scalar_t(0.5) * (m[0].xyz + m[1].xyz + m[2].xyz);
		return v;
	}

	vec3_t getCenter() NBL_CONST_MEMBER_FUNC
	{
		return minCorner + scalar_t(0.5) * (columns[0] + columns[1] + columns[2]);
	}

	vec3_t getVertex(uint32_t i) NBL_CONST_MEMBER_FUNC
	{
		vec3_t p = minCorner;
		if (i & 1u) p += columns[0];
		if (i & 2u) p += columns[1];
		if (i & 4u) p += columns[2];
		return p;
	}

	// Scalar-z specialization: only computes the z component of a corner.
	// Used for clip-mask build where we only need the sign of z.
	scalar_t getVertexZ(uint32_t i) NBL_CONST_MEMBER_FUNC
	{
		scalar_t pz = minCorner.z;
		if (i & 1u) pz += columns[0].z;
		if (i & 2u) pz += columns[1].z;
		if (i & 4u) pz += columns[2].z;
		return pz;
	}

	// Ray-OBB intersection via slab test in OBB-local space.
	// Returns (tMin, tMax, hit). tMin is clamped to 0 if ray starts inside.
	// TODO: not optimized -- precompute inverse columns, handle f~=0 edge case
	struct Intersection
	{
		scalar_t tMin;
		scalar_t tMax;
		bool hit;
	};

	Intersection rayIntersection(vec3_t rayOrigin, vec3_t rayDir) NBL_CONST_MEMBER_FUNC
	{
		// Vector from ray origin to minCorner
		vec3_t delta = rayOrigin - minCorner;

		scalar_t tMin = scalar_t(-1e30);
		scalar_t tMax = scalar_t(1e30);

		// Slab test against each OBB axis
		for (uint32_t i = 0; i < 3; i++)
		{
			vec3_t axis = columns[i];
			scalar_t axisLenSq = dot(axis, axis);
			scalar_t e = dot(axis, delta);
			scalar_t f = dot(axis, rayDir);

			// Project ray onto axis: entry at t where dot = 0, exit where dot = axisLenSq
			scalar_t invF = scalar_t(1.0) / f;
			scalar_t t1 = (-e) * invF;
			scalar_t t2 = (axisLenSq - e) * invF;

			if (t1 > t2)
			{
				scalar_t tmp = t1;
				t1 = t2;
				t2 = tmp;
			}

			tMin = max(tMin, t1);
			tMax = min(tMax, t2);
		}

		Intersection result;
		result.hit = (tMax >= tMin) && (tMax > scalar_t(0));
		result.tMin = max(tMin, scalar_t(0));
		result.tMax = tMax;
		return result;
	}
};

}
}
}

#endif
