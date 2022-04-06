// Copyright (C) 2021-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_SHAPES_OBB_H_INCLUDED_
#define _NBL_CORE_SHAPES_OBB_H_INCLUDED_

#include "nbl/core/decl/Types.h"
#include "nbl/core/math/floatutil.h"
#include "obbox3d.h"

namespace nbl::core
{

//  namespace impl
//  {
//#define uvec2 uint64_t
// include respective OBB glsl shape file
//#undef uvec2
//  }

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
// TODO: redo in terms of `OBB`
//
//  struct CompressedOBB : impl::nbl_glsl_shapes_CompressedOBB_t
//  {
//    CompressedOBB()
//    {
//      const float initMin[] = {FLT_MAX,FLT_MAX,FLT_MAX};
//      minVx18E7S3 = rgb32f_to_rgb18e7s3(initMin);
//      const float initMax[] = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
//      maxVx18E7S3 = rgb32f_to_rgb18e7s3(initMax);
//    }
//    CompressedOBB(const core::OBB& obb)
//    {
//      minVx18E7S3 = rgb32f_to_rgb18e7s3<core::ERD_DOWN>(&obb.bMin.X);
//      maxVx18E7S3 = rgb32f_to_rgb18e7s3<core::ERD_UP>(&obb.bMax.X);
//    }
//
//    inline core::OBB decompress() const
//    {
//      core::OBB obb;
//      reinterpret_cast<rgb32f&>(obb.bMin) = rgb18e7s3_to_rgb32f(minVx18E7S3);
//      reinterpret_cast<rgb32f&>(obb.bMax) = rgb18e7s3_to_rgb32f(maxVx18E7S3);
//      return obb;
//    }
//  };

} // end namespace nbl::video

#endif

