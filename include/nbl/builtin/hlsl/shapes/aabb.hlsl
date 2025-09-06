// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_AABB_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_AABB_INCLUDED_


#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/shapes/util.hlsl"


namespace nbl
{
namespace hlsl
{
namespace shapes
{

template<int16_t D=3, typename Scalar=float32_t>
struct AABB
{
    using scalar_t = Scalar;
    using point_t = vector<Scalar,D>;

    static AABB create()
    {
        AABB retval;
        retval.minVx = promote<point_t>(numeric_limits<Scalar>::max);
        retval.maxVx = promote<point_t>(numeric_limits<Scalar>::lowest);
        return retval;
    }

    //
    void addPoint(const point_t pt)
    {
        minVx = hlsl::min<point_t>(pt,minVx);
        maxVx = hlsl::max<point_t>(pt,maxVx);
    }
    //
    point_t getExtent() NBL_CONST_MEMBER_FUNC
    {
        return maxVx - minVx;
    }

    //
    Scalar getVolume() NBL_CONST_MEMBER_FUNC
    {
        const point_t extent = getExtent();
        return extent.x * extent.y * extent.z;
    }

    // returns the corner of the AABB which has the most positive dot product
    point_t getFarthestPointInFront(const point_t planeNormal) NBL_CONST_MEMBER_FUNC
    {
        return hlsl::mix(maxVx,minVx,planeNormal < promote<point_t>(0.f));
    }

    point_t minVx;
    point_t maxVx;
};

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

namespace util
{
namespace impl
{
template<int16_t D, typename Scalar>
struct intersect_helper<AABB<D,Scalar>>
{
    using type = AABB<D,Scalar>;

    static inline type __call(NBL_CONST_REF_ARG(type) lhs, NBL_CONST_REF_ARG(type) rhs)
    {
        type retval;
        retval.minVx = hlsl::max<type::point_t>(lhs.minVx,rhs.minVx);
        retval.maxVx = hlsl::min<type::point_t>(lhs.maxVx,rhs.maxVx);
        return retval;
    }
};
template<int16_t D, typename Scalar>
struct union_helper<AABB<D,Scalar>>
{
    using type = AABB<D,Scalar>;

    static inline type __call(NBL_CONST_REF_ARG(type) lhs, NBL_CONST_REF_ARG(type) rhs)
    {
        type retval;
        retval.minVx = hlsl::min<type::point_t>(lhs.minVx,rhs.minVx);
        retval.maxVx = hlsl::max<type::point_t>(lhs.maxVx,rhs.maxVx);
        return retval;
    }
};
// without a translation component
template<int16_t D, typename Scalar>
struct transform_helper<AABB<D,Scalar>,matrix<Scalar,D,D> >
{
    using type = AABB<D,Scalar>;
    using matrix_t = matrix<Scalar,D,D>;
    using vector_t = vector<Scalar,D>;

    static inline type __call(NBL_CONST_REF_ARG(matrix_t) lhs, NBL_CONST_REF_ARG(type) rhs)
    {
        // take each column and tweak to get minimum and max
        const matrix_t M_T = hlsl::transpose(lhs);
        vector<bool,D> signM_T[D];
        NBL_UNROLL for (int16_t j=0; j<D; j++)
        NBL_UNROLL for (int16_t i=0; i<D; i++)
            signM_T[j][i] = M_T[j][i]<Scalar(0);

        type retval;
        retval.minVx = M_T[0] * hlsl::mix<vector_t>(rhs.minVx.xxx, rhs.maxVx.xxx, signM_T[0]) + M_T[1] * hlsl::mix<vector_t>(rhs.minVx.yyy, rhs.maxVx.yyy, signM_T[1]) + M_T[2] * hlsl::mix<vector_t>(rhs.minVx.zzz, rhs.maxVx.zzz, signM_T[2]);
        retval.maxVx = M_T[0] * hlsl::mix<vector_t>(rhs.maxVx.xxx, rhs.minVx.xxx, signM_T[0]) + M_T[1] * hlsl::mix<vector_t>(rhs.maxVx.yyy, rhs.minVx.yyy, signM_T[1]) + M_T[2] * hlsl::mix<vector_t>(rhs.maxVx.zzz, rhs.minVx.zzz, signM_T[2]);
        return retval;
    }
};
// affine weird matrix
template<int16_t D, typename Scalar>
struct transform_helper<AABB<D,Scalar>,matrix<Scalar,D,D+1> >
{
    using type = AABB<D,Scalar>;
    using matrix_t = matrix<Scalar,D,D+1>;
    using sub_matrix_t = matrix<Scalar,D,D>;

    static inline type __call(NBL_CONST_REF_ARG(matrix_t) lhs, NBL_CONST_REF_ARG(type) rhs)
    {
        const vector<Scalar,D> translation = hlsl::transpose(lhs)[D];
        type retval = transform_helper<type,sub_matrix_t>::__call(sub_matrix_t(lhs),rhs);
//        retval.minVx += translation;
//        retval.maxVx += translation;
        return retval;
    }
};
}
}

#if 0 // experimental
#include <nbl/builtin/hlsl/format/shared_exp.hlsl>
struct CompressedAABB_t
{
    //
    AABB_t decompress()
    {
        AABB_t retval;
        retval.minVx = decodeRGB18E7S3(minVx18E7S3);
        retval.maxVx = decodeRGB18E7S3(maxVx18E7S3);
        return retval;
    }

    uint2 minVx18E7S3;
    uint2 maxVx18E7S3;
};
#endif


}
}
}

#endif
