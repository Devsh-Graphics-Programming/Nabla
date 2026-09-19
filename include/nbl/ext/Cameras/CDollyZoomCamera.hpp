// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_DOLLY_ZOOM_CAMERA_HPP_
#define _C_DOLLY_ZOOM_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera that preserves subject framing by coupling distance with a derived perspective FOV.
///
/// The rig reuses spherical target-relative manipulation but exposes an additional
/// dynamic-perspective state describing the authored base FOV and the reference
/// distance used to compute the current dolly-zoom FOV.
///
/// Controls: `rotate.y` azimuth and `rotate.x` elevation of the camera around the target, radians, neither
/// clamped; `distance` along the camera-target line, world units, clamped to the distance limits, from which
/// the FOV is derived. The target stays where it is.
class CDollyZoomCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyZoomCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, float baseFov = DefaultBaseFovDeg)
        : base_t(position, target), m_baseFov(baseFov), m_referenceDistance(static_cast<float>(m_orbit.distance))
    {
        updateGimbal();
    }
    ~CDollyZoomCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Return the authored FOV used as the reference value for dolly-zoom evaluation.
    float getBaseFov() const { return m_baseFov; }
    /// @brief Update the authored reference FOV used for dolly-zoom evaluation.
    void setBaseFov(float fov) { m_baseFov = fov; }

    /// @brief Return the reference distance that preserves the authored framing.
    float getReferenceDistance() const { return m_referenceDistance; }
    /// @brief Update the reference distance used by dolly-zoom FOV evaluation.
    void setReferenceDistance(float distance) { m_referenceDistance = distance; }

    /// @brief Evaluate the effective perspective FOV required to preserve subject framing at the current distance.
    float computeDollyFov() const
    {
        const double base = std::tan(hlsl::radians(static_cast<double>(m_baseFov)) * 0.5);
        const double ratio = static_cast<double>(m_referenceDistance) / std::max(m_orbit.distance, MinDistance);
        const double fov = 2.0 * std::atan(base * ratio);
        const double fovDeg = hlsl::degrees(fov);
        return static_cast<float>(std::clamp(fovDeg, static_cast<double>(MinDynamicFovDeg), static_cast<double>(MaxDynamicFovDeg)));
    }

    using base_t::setPose;

    /// @brief Orbit around the current target through the position of `pose`.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        STargetOrbit orbit = {};
        if (!CCameraMathUtilities::tryBuildOrbitFromPosition(m_orbit.target, pose.position, MinDistance, MaxDistance, orbit))
            return false;

        m_orbit = orbit;
        updateGimbal();
        return true;
    }

    virtual uint32_t getAcceptedControls() const override { return AcceptedControls; }
    virtual CameraKind getKind() const override { return CameraKind::DollyZoom; }
    virtual uint32_t getCapabilities() const override { return base_t::getCapabilities() | base_t::DynamicPerspectiveFov; }
    /// @brief Query the current derived FOV produced by the dolly-zoom state.
    virtual bool tryGetDynamicPerspectiveFov(float& outFov) const override
    {
        outFov = computeDollyFov();
        return true;
    }
    /// @brief Query the authored dolly-zoom state used to derive the current dynamic FOV.
    virtual bool tryGetDynamicPerspectiveState(DynamicPerspectiveState& out) const override
    {
        out.baseFov = m_baseFov;
        out.referenceDistance = m_referenceDistance;
        return true;
    }
    /// @brief Replace the authored dolly-zoom state after validating both scalars.
    virtual bool trySetDynamicPerspectiveState(const DynamicPerspectiveState& state) override
    {
        if (!CCameraMathUtilities::isFiniteScalar(state.baseFov) || !CCameraMathUtilities::isFiniteScalar(state.referenceDistance) || state.referenceDistance <= 0.f)
            return false;

        m_baseFov = state.baseFov;
        m_referenceDistance = state.referenceDistance;
        return true;
    }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Dolly Zoom Camera"; }

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::RotateX | ECameraControlAxis::RotateY | ECameraControlAxis::Distance;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        m_orbit.angles += hlsl::float64_t2(controls.rotate.y, controls.rotate.x);
        m_orbit.distance = std::clamp(m_orbit.distance + controls.distance, MinDistance, MaxDistance);

        return updateGimbal();
    }

private:
    static inline constexpr float DefaultBaseFovDeg = 40.0f;
    static inline constexpr float MinDynamicFovDeg = 10.0f;
    static inline constexpr float MaxDynamicFovDeg = 150.0f;

    float m_baseFov = DefaultBaseFovDeg;
    float m_referenceDistance = 1.0f;
};

}

#endif

