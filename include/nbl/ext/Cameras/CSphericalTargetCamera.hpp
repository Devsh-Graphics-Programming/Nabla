// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_SPHERICAL_TARGET_CAMERA_HPP_
#define _C_SPHERICAL_TARGET_CAMERA_HPP_

#include <algorithm>
#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Common base for target-relative cameras, whose whole state is one `STargetOrbit`.
///
/// `updateGimbal()` is the single writer of the gimbal pose and derives it from that state.
/// Derived cameras share the storage and define which parts of the orbit their `manipulate(...)`
/// changes, and how `setPose(...)` projects an authored pose onto it.
class CSphericalTargetCamera : public ICamera
{
public:
    using base_t = ICamera;

    CSphericalTargetCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(), m_orbit{ .target = target, .angles = hlsl::float64_t2(0.0), .distance = MinDistance },
          m_gimbal(SCameraRigPose{ .position = position })
    {
        // a position that coincides with the target has no orbit, so the members keep their fallback
        CCameraMathUtilities::tryBuildOrbitFromPosition(target, position, MinDistance, MaxDistance, m_orbit);
    }
    ~CSphericalTargetCamera() = default;

    inline bool setDistance(const hlsl::float64_t d)
    {
        const auto clamped = std::clamp(d, MinDistance, MaxDistance);
        const bool ok = clamped == d;
        if (m_orbit.distance == clamped)
            return ok;
        m_orbit.distance = clamped;
        updateGimbal();
        return ok;
    }

    inline void target(const hlsl::float64_t3& p)
    {
        if (m_orbit.target == p)
            return;
        m_orbit.target = p;
        updateGimbal();
    }
    inline hlsl::float64_t3 getTarget() const { return m_orbit.target; }

    inline hlsl::float64_t getDistance() const { return m_orbit.distance; }
    /// @brief Return the whole target-relative state backing this camera.
    inline const STargetOrbit& getOrbit() const { return m_orbit; }

    static inline constexpr hlsl::float64_t MinDistance = ICamera::DefaultMinTargetDistance;
    static inline constexpr hlsl::float64_t MaxDistance = ICamera::DefaultMaxTargetDistance;

    virtual uint32_t getCapabilities() const override
    {
        return base_t::SphericalTarget;
    }

    virtual bool tryGetSphericalTargetState(typename base_t::SphericalTargetState& out) const override
    {
        out.target = m_orbit.target;
        out.distance = static_cast<float>(m_orbit.distance);
        out.orbitUv = m_orbit.angles;
        out.minDistance = static_cast<float>(MinDistance);
        out.maxDistance = static_cast<float>(MaxDistance);
        return true;
    }

    virtual bool trySetSphericalTarget(const hlsl::float64_t3& targetPosition) override
    {
        target(targetPosition);
        return true;
    }

    virtual bool trySetSphericalDistance(float distance) override
    {
        return setDistance(static_cast<hlsl::float64_t>(distance));
    }

protected:
    /// @brief Move the target in the view plane of the pose currently committed to the gimbal.
    ///
    /// The camera position is derived from the target, so it follows and the scene slides across the screen.
    /// TODO: the delta is a fixed world-space length, so panning does not track the cursor the way it
    /// does in a DCC. It should scale with `m_orbit.distance`.
    inline void applyPlanarTargetTranslation(const hlsl::float64_t3& deltaTranslation)
    {
        if (!CCameraMathUtilities::hasPlanarDeltaXY(deltaTranslation, static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon)))
            return;

        const auto basis = m_gimbal.getBasis();
        m_orbit.target += CCameraMathUtilities::transformLocalVectorToWorldBasis(
            hlsl::float64_t3(deltaTranslation.x, deltaTranslation.y, 0.0),
            basis.right,
            basis.up,
            basis.forward);
    }

    /// @brief Rebuild the gimbal pose from the current orbit, clamping the stored distance to the legal range.
    /// @return whether the gimbal pose actually changed.
    inline bool updateGimbal()
    {
        SCameraRigPose pose = {};
        if (!CCameraMathUtilities::tryBuildPoseFromOrbit(m_orbit, MinDistance, MaxDistance, pose, &m_orbit.distance))
            return false;

        return m_gimbal.setPose(pose);
    }

    STargetOrbit m_orbit;
    CCameraGimbal m_gimbal;
};

}

#endif
