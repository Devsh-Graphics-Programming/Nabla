#ifndef _C_ORBIT_CAMERA_HPP_
#define _C_ORBIT_CAMERA_HPP_

#include <algorithm>
#include <cmath>
#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera whose state is one `STargetOrbit`.
///
/// Runtime input updates only orbit yaw, orbit pitch, and camera distance.
/// The target position remains unchanged during `manipulate(...)`.
class COrbitCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    COrbitCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbit.distance = std::clamp(hlsl::length(m_orbit.target - position), MinDistance, MaxDistance);
        updateGimbal();
    }
    ~COrbitCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

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

    /// @brief Apply one frame of orbit-angle and distance input around the current target.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const auto deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbit.angles += hlsl::float64_t2(deltaTranslation.y, deltaTranslation.x);
        m_orbit.distance = std::clamp(m_orbit.distance + deltaDistance, MinDistance, MaxDistance);

        return updateGimbal();
    }

    virtual uint32_t getAllowedVirtualEvents() const override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Orbit;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Orbit Camera";
    }

    static inline constexpr hlsl::float64_t MinDistance = base_t::MinDistance;
    static inline constexpr hlsl::float64_t MaxDistance = base_t::MaxDistance;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
};

}

#endif // _C_ORBIT_CAMERA_HPP_
