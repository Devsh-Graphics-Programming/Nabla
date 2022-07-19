// Copyright (C) 2021-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_SHAPES_AABB_H_INCLUDED_
#define _NBL_CORE_SHAPES_AABB_H_INCLUDED_

#include "nbl/core/decl/Types.h"
#include "nbl/core/math/floatutil.h"
#include "aabbox3d.h"

namespace nbl::core
{

namespace impl
{
#define uvec2 uint64_t
#include "nbl/builtin/glsl/shapes/aabb.glsl"
#undef uvec2
}

/* TODO
struct AABB : nbl_glsl_shapes_AABB_t
{
    public:
        //! size in BYTES
        virtual uint64_t getSize() const = 0;

    protected:
        _NBL_INTERFACE_CHILD(IBuffer) {}
};
*/
// TODO: redo in terms of `AABB`
struct NBL_API CompressedAABB : impl::nbl_glsl_shapes_CompressedAABB_t
{
    CompressedAABB()
    {
        const float initMin[] = {FLT_MAX,FLT_MAX,FLT_MAX};
        minVx18E7S3 = rgb32f_to_rgb18e7s3(initMin);
        const float initMax[] = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
        maxVx18E7S3 = rgb32f_to_rgb18e7s3(initMax);
    }
    CompressedAABB(const core::aabbox3df& aabb)
    {
        minVx18E7S3 = rgb32f_to_rgb18e7s3<core::ERD_DOWN>(&aabb.MinEdge.X);
        maxVx18E7S3 = rgb32f_to_rgb18e7s3<core::ERD_UP>(&aabb.MaxEdge.X);
    }

    inline core::aabbox3df decompress() const
    {
        core::aabbox3df aabb;
        reinterpret_cast<rgb32f&>(aabb.MinEdge) = rgb18e7s3_to_rgb32f(minVx18E7S3);
        reinterpret_cast<rgb32f&>(aabb.MaxEdge) = rgb18e7s3_to_rgb32f(maxVx18E7S3);
        return aabb;
    }
};

} // end namespace nbl::video

#endif

