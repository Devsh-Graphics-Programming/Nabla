#ifndef _C_ISOMETRIC_CAMERA_HPP_
#define _C_ISOMETRIC_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera locked to the shared isometric yaw and pitch.
///
/// Translation moves the tracked target in the current view plane while the
/// authored isometric orientation stays fixed. Distance changes are still allowed.
class CIsometricCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CIsometricCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbit.angles = hlsl::float64_t2(IsoYaw, IsoPitch);
        updateGimbal();
    }
    ~CIsometricCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Take the distance of `pose`, holding both angles at the isometric ones.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        STargetOrbit orbit = {};
        if (!CCameraMathUtilities::tryBuildOrbitFromPosition(m_orbit.target, pose.position, MinDistance, MaxDistance, orbit))
            return false;

        orbit.angles = hlsl::float64_t2(IsoYaw, IsoPitch);
        m_orbit = orbit;
        updateGimbal();
        return true;
    }

    /// @brief Apply one frame of planar target translation and distance changes while preserving the fixed isometric angles.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = accumulateVirtualEvents<AllowedVirtualEvents>(virtualEvents);

        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const auto deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbit.angles = hlsl::float64_t2(IsoYaw, IsoPitch);
        m_orbit.distance = std::clamp(m_orbit.distance + deltaDistance, MinDistance, MaxDistance);
        applyPlanarTargetTranslation(deltaTranslation);

        return updateGimbal();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Isometric; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Isometric Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr double IsoYaw = SCameraTargetRelativeRigDefaults::IsometricYawRad;
    static inline const double IsoPitch = SCameraTargetRelativeRigDefaults::IsometricPitchRad;
};

}

#endif
