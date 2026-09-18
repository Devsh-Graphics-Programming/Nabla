#ifndef _C_ISOMETRIC_CAMERA_HPP_
#define _C_ISOMETRIC_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera locked to the shared isometric yaw and pitch.
///
/// Controls: `translate.x` and `translate.y` move the target in the current view plane (x right, y up), world
/// units, so the camera slides with it; `distance` along the camera-target line, world units, clamped to the
/// distance limits. The angles are held at the isometric ones.
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

    virtual uint32_t getAcceptedControls() const override { return AcceptedControls; }
    virtual CameraKind getKind() const override { return CameraKind::Isometric; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Isometric Camera"; }

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::TranslateX | ECameraControlAxis::TranslateY | ECameraControlAxis::Distance;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        m_orbit.angles = hlsl::float64_t2(IsoYaw, IsoPitch);
        m_orbit.distance = std::clamp(m_orbit.distance + controls.distance, MinDistance, MaxDistance);
        applyPlanarTargetTranslation(controls.translate);

        return updateGimbal();
    }

private:
    static inline constexpr double IsoYaw = SCameraTargetRelativeRigDefaults::IsometricYawRad;
    static inline const double IsoPitch = SCameraTargetRelativeRigDefaults::IsometricPitchRad;
};

}

#endif
