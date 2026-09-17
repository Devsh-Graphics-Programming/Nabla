#ifndef _C_DOLLY_CAMERA_HPP_
#define _C_DOLLY_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera that translates the target in the full local camera basis.
///
/// Translation uses the current right/up/forward basis. Rotation updates orbit
/// yaw and pitch while the camera pose is rebuilt from the maintained
/// target-relative offset.
class CDollyCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbit.angles.y = std::clamp(m_orbit.angles.y, MinPitch, MaxPitch);
        updateGimbal();
    }
    ~CDollyCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Orbit around the target through the position of `pose`, under the dolly pitch limit.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        STargetOrbit orbit = {};
        if (!CCameraMathUtilities::tryBuildOrbitFromPosition(m_orbit.target, pose.position, MinDistance, MaxDistance, orbit))
            return false;

        orbit.angles.y = std::clamp(orbit.angles.y, MinPitch, MaxPitch);
        m_orbit = orbit;
        updateGimbal();
        return true;
    }

    /// @brief Apply one frame of local-frame dolly translation plus orbit rotation.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = accumulateVirtualEvents<AllowedVirtualEvents>(virtualEvents);

        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);

        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        // the dolly moves the target through the full committed basis, forward component included
        // TODO: like the planar pan in the base rig, this delta is a fixed world-space length and should scale with distance
        const auto basis = m_gimbal.getBasis();
        const auto delta = CCameraMathUtilities::transformLocalVectorToWorldBasis(deltaTranslation, basis.right, basis.up, basis.forward);

        m_orbit.target += delta;
        m_orbit.angles.x += deltaRotation.y;
        m_orbit.angles.y = std::clamp(m_orbit.angles.y + deltaRotation.x, MinPitch, MaxPitch);

        return updateGimbal();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Dolly; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Dolly Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::DollyPitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
