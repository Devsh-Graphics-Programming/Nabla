// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PROJECTION_UTILITIES_HPP_
#define _C_CAMERA_PROJECTION_UTILITIES_HPP_

#include "IPlanarProjection.hpp"

namespace nbl::core
{

struct CCameraProjectionUtilities final
{
    /// @brief Apply a camera-provided dynamic perspective FOV to one planar projection entry.
    static inline bool syncDynamicPerspectiveProjection(ICamera* camera, IPlanarProjection::CProjection& projection)
    {
        if (!camera)
            return false;

        const auto& params = projection.getParameters();
        if (params.m_type != IPlanarProjection::CProjection::Perspective)
            return false;

        float dynamicFov = 0.0f;
        if (!camera->tryGetDynamicPerspectiveFov(dynamicFov))
            return false;

        projection.setPerspective(params.m_zNear, params.m_zFar, dynamicFov);
        return true;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_PROJECTION_UTILITIES_HPP_
