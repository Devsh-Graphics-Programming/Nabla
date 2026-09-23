// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/ICameraWithProjections.hpp"

namespace nbl::ext::cameras
{

bool ICameraWithProjections::setCamera(core::smart_refctd_ptr<ICamera>&& camera)
{
    if (!camera)
        return false;

    m_camera = std::move(camera);
    return true;
}

ICamera* ICameraWithProjections::getCamera()
{
    return m_camera.get();
}

hlsl::float64_t4x4 ICameraWithProjections::getViewMatrix() const
{
    return hlsl::math::linalg::promote_affine<4,4,3,4>(m_camera->getGimbal().getViewMatrixLH());
}

hlsl::float64_t4x4 ICameraWithProjections::getProjectionMatrix(const uint32_t projectionIx) const
{
    return getProjection(projectionIx).getProjectionMatrix();
}

hlsl::float64_t4x4 ICameraWithProjections::getViewProjectionMatrix(const uint32_t projectionIx) const
{
    return hlsl::mul(getProjectionMatrix(projectionIx), getViewMatrix());
}

} // namespace nbl::ext::cameras
