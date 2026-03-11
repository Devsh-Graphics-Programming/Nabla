// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_AABB_ACCUMULATOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_AABB_ACCUMULATOR_INCLUDED_


#include "nbl/builtin/hlsl/shapes/aabb.hlsl"
#include "nbl/builtin/hlsl/array_accessors.hlsl"
#include "nbl/builtin/hlsl/concepts/vector.hlsl"


namespace nbl
{
namespace hlsl
{
namespace shapes
{
namespace util
{

template<typename Scalar = float32_t>
struct AABBAccumulator3
{
    using scalar_t = Scalar;
    using aabb_t = AABB<3, Scalar>;
    using point_t = typename aabb_t::point_t;

    static AABBAccumulator3 create()
    {
        AABBAccumulator3 retval;
        retval.value = aabb_t::create();
        return retval;
    }

    bool empty() NBL_CONST_MEMBER_FUNC
    {
        return
            value.minVx.x > value.maxVx.x ||
            value.minVx.y > value.maxVx.y ||
            value.minVx.z > value.maxVx.z;
    }

    void addPoint(NBL_CONST_REF_ARG(point_t) pt)
    {
        value.addPoint(pt);
    }

    void addXYZ(const Scalar x, const Scalar y, const Scalar z)
    {
        point_t pt = point_t(x, y, z);
        value.addPoint(pt);
    }

    aabb_t value;
};

template<typename Scalar>
inline AABBAccumulator3<Scalar> createAABBAccumulator()
{
    return AABBAccumulator3<Scalar>::create();
}

template<typename Scalar>
inline void extendAABBAccumulator(NBL_REF_ARG(AABBAccumulator3<Scalar>) aabb, const Scalar x, const Scalar y, const Scalar z)
{
    aabb.addXYZ(x, y, z);
}

template<typename Scalar, typename Point NBL_FUNC_REQUIRES(concepts::Vectorial<Point> && (vector_traits<Point>::Dimension >= 3))
inline void extendAABBAccumulator(NBL_REF_ARG(AABBAccumulator3<Scalar>) aabb, NBL_CONST_REF_ARG(Point) pt)
{
    array_get<Point, typename vector_traits<Point>::scalar_type> getter;
    typename AABBAccumulator3<Scalar>::point_t converted = typename AABBAccumulator3<Scalar>::point_t(
        Scalar(getter(pt, 0)),
        Scalar(getter(pt, 1)),
        Scalar(getter(pt, 2))
    );
    aabb.addPoint(converted);
}

template<int16_t DstD, typename DstScalar, int16_t SrcD, typename SrcScalar NBL_FUNC_REQUIRES(DstD >= 3 && SrcD >= 3)
inline bool assignAABB(NBL_REF_ARG(AABB<DstD, DstScalar>) dst, NBL_CONST_REF_ARG(AABB<SrcD, SrcScalar>) src)
{
    array_set<typename AABB<DstD, DstScalar>::point_t, DstScalar> setter;
    array_get<typename AABB<SrcD, SrcScalar>::point_t, SrcScalar> getter;

    if (
        getter(src.minVx, 0) > getter(src.maxVx, 0) ||
        getter(src.minVx, 1) > getter(src.maxVx, 1) ||
        getter(src.minVx, 2) > getter(src.maxVx, 2))
        return false;

    dst = AABB<DstD, DstScalar>::create();
    NBL_UNROLL for (int16_t i = 0; i < 3; ++i)
    {
        setter(dst.minVx, i, DstScalar(getter(src.minVx, i)));
        setter(dst.maxVx, i, DstScalar(getter(src.maxVx, i)));
    }
    NBL_UNROLL for (int16_t i = 3; i < DstD; ++i)
    {
        setter(dst.minVx, i, DstScalar(0));
        setter(dst.maxVx, i, DstScalar(0));
    }
    return true;
}

template<int16_t D, typename DstScalar, typename SrcScalar NBL_FUNC_REQUIRES(D >= 3)
inline bool assignAABBFromAccumulator(NBL_REF_ARG(AABB<D, DstScalar>) dst, NBL_CONST_REF_ARG(AABBAccumulator3<SrcScalar>) aabb)
{
    if (aabb.empty())
        return false;

    return assignAABB(dst, aabb.value);
}

}
}
}
}

#endif
