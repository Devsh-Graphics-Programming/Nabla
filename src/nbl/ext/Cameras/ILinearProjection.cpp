// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/ILinearProjection.hpp"

namespace nbl::core
{

ILinearProjection::CProjection::CProjection() : CProjection(projection_matrix_t(1))
{
}

ILinearProjection::CProjection::CProjection(const projection_matrix_t& matrix)
{
    setProjectionMatrix(matrix);
}

void ILinearProjection::CProjection::setProjectionMatrix(const projection_matrix_t& matrix)
{
    m_projectionMatrix = matrix;
    const auto det = hlsl::determinant(m_projectionMatrix);

    m_isProjectionSingular = !det;

    if (m_isProjectionSingular)
    {
        m_isProjectionLeftHanded = std::nullopt;
        m_invProjectionMatrix = std::nullopt;
    }
    else
    {
        m_isProjectionLeftHanded = det < 0.0;
        m_invProjectionMatrix = hlsl::inverse(m_projectionMatrix);
    }
}

bool ILinearProjection::setCamera(core::smart_refctd_ptr<ICamera>&& camera)
{
    if (!camera)
        return false;

    m_camera = std::move(camera);
    return true;
}

ICamera* ILinearProjection::getCamera()
{
    return m_camera.get();
}

ILinearProjection::concatenated_matrix_t ILinearProjection::getMV(const model_matrix_t& model) const
{
    const auto& view = m_camera->getGimbal().getViewMatrix();
    return hlsl::mul(
        hlsl::CCameraMathUtilities::promoteAffine3x4To4x4(view),
        hlsl::CCameraMathUtilities::promoteAffine3x4To4x4(model));
}

ILinearProjection::concatenated_matrix_t ILinearProjection::getMVP(const CProjection& projection, const model_matrix_t& model) const
{
    return getMVP(projection, getMV(model));
}

ILinearProjection::concatenated_matrix_t ILinearProjection::getMVP(const CProjection& projection, const concatenated_matrix_t& mv) const
{
    return hlsl::mul(projection.getProjectionMatrix(), mv);
}

ILinearProjection::inv_concatenated_matrix_t ILinearProjection::getMVInverse(const model_matrix_t& model) const
{
    const auto mv = getMV(model);
    if (const auto det = hlsl::determinant(mv); det)
        return hlsl::inverse(mv);
    return std::nullopt;
}

ILinearProjection::inv_concatenated_matrix_t ILinearProjection::getMVPInverse(const CProjection& projection, const model_matrix_t& model) const
{
    const auto mvp = getMVP(projection, model);
    if (const auto det = hlsl::determinant(mvp); det)
        return hlsl::inverse(mvp);
    return std::nullopt;
}

} // namespace nbl::core
