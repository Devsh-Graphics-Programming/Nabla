// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "CCameraGoalSolver.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraKindUtilities.hpp"

namespace nbl::core
{

/// @brief Reusable tracked-target and follow helpers.
///
/// The tracked subject owns its own gimbal. Follow code reads that pose and
/// maps one camera plus one tracked target into a `CCameraGoal`.
class CTrackedTarget
{
public:
    using gimbal_t = ICamera::CGimbal;

    /// @brief Construct a tracked target from an initial pose and optional identifier.
    CTrackedTarget(
        const hlsl::float64_t3& position = hlsl::float64_t3(0.0),
        const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>(),
        std::string identifier = "Follow Target");

    /// @brief Return the stable human-readable identifier of the tracked target.
    inline const std::string& getIdentifier() const { return m_identifier; }
    /// @brief Return read-only access to the tracked target gimbal.
    inline const gimbal_t& getGimbal() const { return m_gimbal; }
    /// @brief Return mutable access to the tracked target gimbal.
    inline gimbal_t& getGimbal() { return m_gimbal; }

    /// @brief Replace the tracked target pose in world space.
    void setPose(const hlsl::float64_t3& position, const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation);

    /// @brief Replace only the tracked target position.
    void setPosition(const hlsl::float64_t3& position);

    /// @brief Replace only the tracked target orientation.
    void setOrientation(const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation);

    /// @brief Replace the tracked target pose from a rigid transform matrix when possible.
    bool trySetFromTransform(const hlsl::float64_t4x4& transform);

private:
    std::string m_identifier;
    gimbal_t m_gimbal;
};

/// @brief Follow policy layered on top of a tracked target gimbal.
///
/// Each mode defines how tracked-target motion updates the camera:
///
/// - `OrbitTarget` rewrites target-relative camera state so the tracked target becomes the camera target
/// - `LookAtTarget` preserves camera position and rebuilds orientation toward the tracked target
/// - `KeepWorldOffset` places the camera at `trackedTarget.position + worldOffset` and looks at the target
/// - `KeepLocalOffset` transforms `localOffset` by the tracked-target local frame and looks at the target
///
/// The tracked target provides pose data. The camera reads that data and does
/// not own the tracked subject.
enum class ECameraFollowMode : uint8_t
{
    Disabled,
    OrbitTarget,
    LookAtTarget,
    KeepWorldOffset,
    KeepLocalOffset
};

/// @brief Reusable follow configuration interpreted against a tracked target gimbal.
/// `worldOffset` and `localOffset` are only meaningful for their matching offset-based modes.
struct SCameraFollowConfig
{
    /// @brief Whether follow should be applied at all.
    bool enabled = false;
    /// @brief Follow policy used when the configuration is enabled.
    ECameraFollowMode mode = ECameraFollowMode::OrbitTarget;
    /// @brief World-space offset preserved by `KeepWorldOffset`.
    hlsl::float64_t3 worldOffset = hlsl::float64_t3(0.0);
    /// @brief Target-local offset preserved by `KeepLocalOffset`.
    hlsl::float64_t3 localOffset = hlsl::float64_t3(0.0);
};

/// @brief Shared policy helpers for tracked-target follow.
///
/// The helpers decide which follow modes lock the view, which ones move the
/// camera, how offsets are captured, and how a tracked target is translated into
/// a `CCameraGoal` that can then be applied through the shared goal solver.
struct CCameraFollowUtilities final
{
    /// @brief Return whether the follow mode rebuilds camera orientation toward the tracked target.
    static inline constexpr bool cameraFollowModeLocksViewToTarget(const ECameraFollowMode mode)
    {
        switch (mode)
        {
            case ECameraFollowMode::OrbitTarget:
            case ECameraFollowMode::LookAtTarget:
            case ECameraFollowMode::KeepWorldOffset:
            case ECameraFollowMode::KeepLocalOffset:
                return true;
            default:
                return false;
        }
    }

    /// @brief Return whether the follow mode moves the camera world position together with the target.
    static inline constexpr bool cameraFollowModeMovesCameraPosition(const ECameraFollowMode mode)
    {
        switch (mode)
        {
            case ECameraFollowMode::OrbitTarget:
            case ECameraFollowMode::KeepWorldOffset:
            case ECameraFollowMode::KeepLocalOffset:
                return true;
            default:
                return false;
        }
    }

    /// @brief Return whether the follow mode preserves the current camera world position.
    static inline constexpr bool cameraFollowModeKeepsCameraWorldPosition(const ECameraFollowMode mode)
    {
        return mode == ECameraFollowMode::LookAtTarget;
    }

    /// @brief Return whether the follow mode interprets `worldOffset`.
    static inline constexpr bool cameraFollowModeUsesWorldOffset(const ECameraFollowMode mode)
    {
        return mode == ECameraFollowMode::KeepWorldOffset;
    }

    /// @brief Return whether the follow mode interprets `localOffset`.
    static inline constexpr bool cameraFollowModeUsesLocalOffset(const ECameraFollowMode mode)
    {
        return mode == ECameraFollowMode::KeepLocalOffset;
    }

    /// @brief Return whether the follow mode needs the tracked target local frame.
    static inline constexpr bool cameraFollowModeUsesTrackedTargetLocalFrame(const ECameraFollowMode mode)
    {
        return mode == ECameraFollowMode::KeepLocalOffset;
    }

    /// @brief Return whether the follow mode requires a captured offset before it can be replayed.
    static inline constexpr bool cameraFollowModeUsesCapturedOffset(const ECameraFollowMode mode)
    {
        return cameraFollowModeUsesWorldOffset(mode) || cameraFollowModeUsesLocalOffset(mode);
    }

    /// @brief Return the shared default follow mode for one camera kind.
    static inline constexpr ECameraFollowMode getDefaultCameraFollowMode(const ICamera::CameraKind kind)
    {
        switch (kind)
        {
            case ICamera::CameraKind::Orbit:
            case ICamera::CameraKind::Arcball:
            case ICamera::CameraKind::Turntable:
            case ICamera::CameraKind::TopDown:
            case ICamera::CameraKind::Isometric:
            case ICamera::CameraKind::DollyZoom:
            case ICamera::CameraKind::Path:
                return ECameraFollowMode::OrbitTarget;
            case ICamera::CameraKind::Chase:
            case ICamera::CameraKind::Dolly:
                return ECameraFollowMode::KeepLocalOffset;
            default:
                return ECameraFollowMode::Disabled;
        }
    }

    /// @brief Build the shared default follow configuration for one camera kind.
    static inline constexpr SCameraFollowConfig makeDefaultFollowConfig(const ICamera::CameraKind kind)
    {
        const auto mode = getDefaultCameraFollowMode(kind);
        return {
            .enabled = mode != ECameraFollowMode::Disabled,
            .mode = mode
        };
    }

    /// @brief Build the shared default follow configuration for one concrete camera instance.
    static inline constexpr SCameraFollowConfig makeDefaultFollowConfig(const ICamera* const camera)
    {
        return camera ? makeDefaultFollowConfig(camera->getKind()) : SCameraFollowConfig{};
    }

    /// @brief Transform a tracked-target local offset into world space.
    static hlsl::float64_t3 transformFollowLocalOffset(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& localOffset);

    /// @brief Project a world-space offset into the tracked target local frame.
    static hlsl::float64_t3 projectFollowWorldOffsetToLocal(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& worldOffset);

    /// @brief Build a look-at orientation that points from `position` toward the tracked target.
    static bool buildFollowLookAtOrientation(
        const hlsl::float64_t3& position,
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& preferredUp,
        hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation);

    /// @brief Capture world-space and target-local follow offsets from the current camera pose.
    static bool captureFollowOffsetsFromCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        SCameraFollowConfig& ioConfig);

    /// @brief Measure the angular lock error between a camera forward axis and a tracked target.
    static bool tryComputeFollowTargetLockMetrics(
        const ICamera::CGimbal& cameraGimbal,
        const CTrackedTarget& trackedTarget,
        float& outAngleDeg,
        double* outDistance = nullptr);

    static bool tryBuildFollowPositionGoal(
        ICamera* camera,
        CCameraGoal& outGoal,
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const hlsl::float64_t3& preferredUp);

    static bool tryBuildFollowGoal(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& config,
        CCameraGoal& outGoal);

    static CCameraGoalSolver::SApplyResult applyFollowToCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& config,
        CCameraGoal* outGoal = nullptr);
};

} // namespace nbl::core

#endif // _C_CAMERA_FOLLOW_UTILITIES_HPP_

