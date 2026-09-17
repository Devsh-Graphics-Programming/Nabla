#ifndef _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_
#define _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_

#include <limits>

#include "SCameraTypes.hpp"
#include "CCameraVirtualEventUtilities.hpp"

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

/// @brief Mapping policy describing how a target-relative delta is converted into virtual events.
struct SCameraTargetRelativeEventPolicy final
{
    bool translateOrbit = false;
    bool allowYaw = true;
    bool allowPitch = true;
    SCameraVirtualEventAxisBinding distanceBinding = {
        CVirtualGimbalEvent::MoveForward,
        CVirtualGimbalEvent::MoveBackward
    };
};

/// @brief Default constants and event policies used by target-relative rigs.
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

    static inline constexpr SCameraTargetRelativeEventPolicy OrbitTranslatePolicy = {
        .translateOrbit = true
    };
    static inline constexpr SCameraTargetRelativeEventPolicy RotateDistancePolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = true
    };
    static inline constexpr SCameraTargetRelativeEventPolicy TopDownPolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = false
    };
    static inline constexpr SCameraTargetRelativeEventPolicy IsometricPolicy = {
        .translateOrbit = false,
        .allowYaw = false,
        .allowPitch = false
    };
    static inline constexpr SCameraTargetRelativeEventPolicy DollyPolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = true,
        .distanceBinding = {
            CVirtualGimbalEvent::None,
            CVirtualGimbalEvent::None
        }
    };
    static inline constexpr SCameraTargetRelativeEventPolicy ChasePolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = true,
        .distanceBinding = {
            CVirtualGimbalEvent::MoveUp,
            CVirtualGimbalEvent::MoveDown
        }
    };
};

/// @brief Helpers for converting between target-relative state, pose, basis, and virtual-event deltas.
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

    static inline void appendTargetRelativeDeltaEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraTargetRelativeDelta& delta,
        const double angularDenominator,
        const double angularToleranceDeg,
        const double distanceDenominator,
        const double distanceTolerance,
        const SCameraTargetRelativeEventPolicy& policy)
    {
        if (policy.translateOrbit)
        {
            CCameraVirtualEventUtilities::appendAngularAxisEvents(
                events,
                delta.orbitVector(),
                hlsl::float64_t3(angularDenominator),
                hlsl::float64_t3(angularToleranceDeg, angularToleranceDeg, std::numeric_limits<hlsl::float64_t>::infinity()),
                {{
                    { CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft },
                    { CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown },
                    { CVirtualGimbalEvent::None, CVirtualGimbalEvent::None }
                }});
        }
        else
        {
            if (policy.allowYaw)
            {
                CCameraVirtualEventUtilities::appendAngularDeltaEvent(
                    events,
                    delta.orbitUv.x,
                    angularDenominator,
                    angularToleranceDeg,
                    CVirtualGimbalEvent::PanRight,
                    CVirtualGimbalEvent::PanLeft);
            }
            if (policy.allowPitch)
            {
                CCameraVirtualEventUtilities::appendAngularDeltaEvent(
                    events,
                    delta.orbitUv.y,
                    angularDenominator,
                    angularToleranceDeg,
                    CVirtualGimbalEvent::TiltUp,
                    CVirtualGimbalEvent::TiltDown);
            }
        }

        if (policy.distanceBinding.positive != CVirtualGimbalEvent::None &&
            policy.distanceBinding.negative != CVirtualGimbalEvent::None)
        {
            CCameraVirtualEventUtilities::appendScaledVirtualEvent(
                events,
                delta.distance,
                distanceDenominator,
                distanceTolerance,
                policy.distanceBinding.positive,
                policy.distanceBinding.negative);
        }
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_

