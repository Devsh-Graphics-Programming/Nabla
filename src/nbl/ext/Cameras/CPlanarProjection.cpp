// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <cassert>

#include "nbl/ext/Cameras/CPlanarProjection.hpp"
#include "nbl/builtin/hlsl/math/thin_lens_projection.hlsl"

namespace nbl::ext::cameras
{

CPlanarProjection::CPlanarProjection()
{
    setProjectionMatrix(hlsl::float64_t4x4(1));
}

CPlanarProjection CPlanarProjection::createPerspective(const float zNear, const float zFar, const float fov)
{
    CPlanarProjection output;
    output.setPerspective(zNear, zFar, fov);
    return output;
}

CPlanarProjection CPlanarProjection::createOrthographic(const float zNear, const float zFar, const float orthoWidth)
{
    CPlanarProjection output;
    output.setOrthographic(zNear, zFar, orthoWidth);
    return output;
}

CPlanarProjection CPlanarProjection::create(const SParameters& parameters)
{
    assert(parameters.kind == EKind::Perspective || parameters.kind == EKind::Orthographic);

    CPlanarProjection output;
    output.m_parameters = parameters;
    return output;
}

CPlanarProjection CPlanarProjection::create(const hlsl::float64_t4x4& matrix)
{
    CPlanarProjection output;
    output.m_parameters.kind = EKind::Custom;
    output.setProjectionMatrix(matrix);
    return output;
}

void CPlanarProjection::setProjectionMatrix(const hlsl::float64_t4x4& matrix)
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

void CPlanarProjection::update(const bool leftHanded, const float aspectRatio)
{
    switch (m_parameters.kind)
    {
        case EKind::Perspective:
        {
            const auto& fov = m_parameters.perspective.fov;

            if (leftHanded)
                setProjectionMatrix(hlsl::buildProjectionMatrixPerspectiveFovLH<hlsl::float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.zNear, m_parameters.zFar));
            else
                setProjectionMatrix(hlsl::buildProjectionMatrixPerspectiveFovRH<hlsl::float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.zNear, m_parameters.zFar));
        } break;

        case EKind::Orthographic:
        {
            const auto& orthoW = m_parameters.orthographic.orthoWidth;
            const auto viewHeight = orthoW / aspectRatio;

            if (leftHanded)
                setProjectionMatrix(hlsl::buildProjectionMatrixOrthoLH<hlsl::float64_t>(orthoW, viewHeight, m_parameters.zNear, m_parameters.zFar));
            else
                setProjectionMatrix(hlsl::buildProjectionMatrixOrthoRH<hlsl::float64_t>(orthoW, viewHeight, m_parameters.zNear, m_parameters.zFar));
        } break;

        default:
            break;
    }
}

void CPlanarProjection::setPerspective(const float zNear, const float zFar, const float fov)
{
    m_parameters.kind = EKind::Perspective;
    m_parameters.perspective.fov = fov;
    m_parameters.zNear = zNear;
    m_parameters.zFar = zFar;
}

void CPlanarProjection::setOrthographic(const float zNear, const float zFar, const float orthoWidth)
{
    m_parameters.kind = EKind::Orthographic;
    m_parameters.orthographic.orthoWidth = orthoWidth;
    m_parameters.zNear = zNear;
    m_parameters.zFar = zFar;
}

} // namespace nbl::ext::cameras
