#ifndef _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_
#define _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_

#include <limits>

#include "SCameraTypes.hpp"
#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Pose reconstructed from an `STargetOrbit`, carrying the distance that was actually applied.
struct SCameraTargetRelativePose final : SCameraRigPose
{
    hlsl::float64_t appliedDistance = static_cast<hlsl::float64_t>(ICamera::DefaultMinTargetDistance);
};

/// @brief Delta between current spherical target state and canonical target-relative goal.
struct SCameraTargetRelativeDelta final
{
    hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
    double distance = 0.0;

    inline hlsl::float64_t3 orbitVector() const
    {
        return hlsl::float64_t3(orbitUv.y, orbitUv.x, 0.0);
    }
};

/// @brief Default constants used by target-relative rigs.
struct SCameraTargetRelativeRigDefaults final
{
    static constexpr float InitialDistance = 1.0f;
    static constexpr double ArcballPitchLimitRad = SCameraViewRigDefaults::ArcballPitchLimitRad;
    static constexpr double TurntablePitchLimitRad = SCameraViewRigDefaults::TurntablePitchLimitRad;
    static constexpr double ChaseMaxPitchRad = SCameraViewRigDefaults::ChaseMaxPitchRad;
    static constexpr double ChaseMinPitchRad = SCameraViewRigDefaults::ChaseMinPitchRad;
    static constexpr double DollyPitchLimitRad = SCameraViewRigDefaults::DollyPitchLimitRad;
    static constexpr double TopDownPitchRad = SCameraViewRigDefaults::TopDownPitchRad;
    static constexpr double IsometricYawRad = SCameraViewRigDefaults::IsometricYawRad;
    static inline const double IsometricPitchRad = SCameraViewRigDefaults::IsometricPitchRad;
};

/// @brief Helpers for converting between target-relative state and a desired orbit.
struct CCameraTargetRelativeUtilities final
{
    static inline SCameraTargetRelativeDelta buildTargetRelativeDelta(
        const ICamera::SphericalTargetState& currentState,
        const STargetOrbit& desiredOrbit)
    {
        return {
            .orbitUv = hlsl::float64_t2(
                CCameraMathUtilities::wrapAngleRad(desiredOrbit.angles.x - currentState.orbitUv.x),
                CCameraMathUtilities::wrapAngleRad(desiredOrbit.angles.y - currentState.orbitUv.y)),
            .distance = desiredOrbit.distance - static_cast<hlsl::float64_t>(currentState.distance)
        };
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_

