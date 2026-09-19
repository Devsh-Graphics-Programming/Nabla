// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_C_PLANAR_PROJECTION_HPP_
#define _NBL_C_PLANAR_PROJECTION_HPP_

#include <optional>

// the HLSL headers below expect `<algorithm>` and `core::reciprocal` to be declared already
#include "nbl/core/math/glslFunctions.h"
#include "IProjection.hpp"

namespace nbl::ext::cameras
{

/// @brief One linear projection onto an image plane, held by value.
///
/// Perspective and orthographic projections store their authored parameters and rebuild the
/// matrix from them in `update`, which takes the viewport's handedness and aspect ratio.
/// Setting parameters does not touch the matrix until the next `update`.
/// A `Custom` projection wraps a caller-provided matrix and has no parameters.
class CPlanarProjection : public IProjection
{
public:
    /// @brief How the projection matrix is produced.
    enum class EKind : uint8_t
    {
        Perspective,
        Orthographic,

        /// @brief Arbitrary caller-provided matrix, for example oblique or shear. `update` leaves it unchanged.
        Custom,

        Count
    };

    /// @brief Authored parameters of a perspective or orthographic projection.
    struct SParameters
    {
        struct SPerspective
        {
            /// @brief Field of view in degrees.
            float fov;
        };

        struct SOrthographic
        {
            float orthoWidth;
        };

        EKind kind = EKind::Perspective;

        union
        {
            SPerspective perspective = { .fov = 60.f };
            SOrthographic orthographic;
        };

        float zNear = 0.1f;
        float zFar = 100.f;
    };

    /// @brief Create a perspective projection. `fov` is in degrees.
    static CPlanarProjection createPerspective(float zNear = 0.1f, float zFar = 100.f, float fov = 60.f);

    /// @brief Create an orthographic projection.
    static CPlanarProjection createOrthographic(float zNear = 0.1f, float zFar = 100.f, float orthoWidth = 10.f);

    /// @brief Create a perspective or orthographic projection from authored parameters. `kind` must not be `Custom`.
    static CPlanarProjection create(const SParameters& parameters);

    /// @brief Create a `Custom` projection wrapping an arbitrary matrix.
    static CPlanarProjection create(const hlsl::float64_t4x4& matrix);

    /// @brief Switch to perspective and store its parameters. `fov` is in degrees.
    void setPerspective(float zNear = 0.1f, float zFar = 100.f, float fov = 60.f);

    /// @brief Switch to orthographic and store its parameters.
    void setOrthographic(float zNear = 0.1f, float zFar = 100.f, float orthoWidth = 10.f);

    /// @brief Rebuild the matrix from the stored parameters. Does nothing for `Custom`.
    void update(bool leftHanded, float aspectRatio);

    /// @brief Return the authored parameters.
    inline const SParameters& getParameters() const { return m_parameters; }

    /// @brief Returns P (Projection matrix)
    inline const hlsl::float64_t4x4& getProjectionMatrix() const { return m_projectionMatrix; }

    /// @brief Returns P⁻¹ (Inverse of Projection matrix) *if it exists*
    inline const std::optional<hlsl::float64_t4x4>& getInvProjectionMatrix() const { return m_invProjectionMatrix; }

    inline const std::optional<bool>& isProjectionLeftHanded() const { return m_isProjectionLeftHanded; }
    inline bool isProjectionSingular() const { return m_isProjectionSingular; }

    virtual ProjectionType getProjectionType() const override { return ProjectionType::Planar; }

    virtual void project(const hlsl::float64_t4& vecToProjectionSpace, hlsl::float64_t4& output) const override
    {
        output = hlsl::mul(m_projectionMatrix, vecToProjectionSpace);
    }

    virtual bool unproject(const hlsl::float64_t4& vecFromProjectionSpace, hlsl::float64_t4& output) const override
    {
        if (m_isProjectionSingular)
            return false;

        output = hlsl::mul(m_invProjectionMatrix.value(), vecFromProjectionSpace);

        return true;
    }

private:
    CPlanarProjection();

    /// @brief Replace the projection matrix and rebuild cached handedness and inverse information.
    void setProjectionMatrix(const hlsl::float64_t4x4& matrix);

    SParameters m_parameters;
    hlsl::float64_t4x4 m_projectionMatrix;
    std::optional<hlsl::float64_t4x4> m_invProjectionMatrix;
    std::optional<bool> m_isProjectionLeftHanded;
    bool m_isProjectionSingular;
};

} // namespace nbl::ext::cameras

#endif // _NBL_C_PLANAR_PROJECTION_HPP_
