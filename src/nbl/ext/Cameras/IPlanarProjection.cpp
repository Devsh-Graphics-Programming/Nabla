// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/IPlanarProjection.hpp"

namespace nbl::core
{

void IPlanarProjection::CProjection::update(const bool leftHanded, const float aspectRatio)
{
    switch (m_parameters.m_type)
    {
        case Perspective:
        {
            const auto& fov = m_parameters.m_planar.perspective.fov;

            if (leftHanded)
                base_t::setProjectionMatrix(hlsl::buildProjectionMatrixPerspectiveFovLH<hlsl::float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
            else
                base_t::setProjectionMatrix(hlsl::buildProjectionMatrixPerspectiveFovRH<hlsl::float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
        } break;

        case Orthographic:
        {
            const auto& orthoW = m_parameters.m_planar.orthographic.orthoWidth;
            const auto viewHeight = orthoW / aspectRatio;

            if (leftHanded)
                base_t::setProjectionMatrix(hlsl::buildProjectionMatrixOrthoLH<hlsl::float64_t>(orthoW, viewHeight, m_parameters.m_zNear, m_parameters.m_zFar));
            else
                base_t::setProjectionMatrix(hlsl::buildProjectionMatrixOrthoRH<hlsl::float64_t>(orthoW, viewHeight, m_parameters.m_zNear, m_parameters.m_zFar));
        } break;
    }
}

void IPlanarProjection::CProjection::setPerspective(const float zNear, const float zFar, const float fov)
{
    m_parameters.m_type = Perspective;
    m_parameters.m_planar.perspective.fov = fov;
    m_parameters.m_zNear = zNear;
    m_parameters.m_zFar = zFar;
}

void IPlanarProjection::CProjection::setOrthographic(const float zNear, const float zFar, const float orthoWidth)
{
    m_parameters.m_type = Orthographic;
    m_parameters.m_planar.orthographic.orthoWidth = orthoWidth;
    m_parameters.m_zNear = zNear;
    m_parameters.m_zFar = zFar;
}

} // namespace nbl::core
