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
        std::string identifier = "Follow Target")
        : m_identifier(std::move(identifier)),
        m_gimbal(gimbal_t::base_t::SCreationParameters{ .position = position, .orientation = orientation })
    {
        m_gimbal.updateView();
    }

    /// @brief Return the stable human-readable identifier of the tracked target.
    inline const std::string& getIdentifier() const { return m_identifier; }
    /// @brief Return read-only access to the tracked target gimbal.
    inline const gimbal_t& getGimbal() const { return m_gimbal; }
    /// @brief Return mutable access to the tracked target gimbal.
    inline gimbal_t& getGimbal() { return m_gimbal; }

    /// @brief Replace the tracked target pose in world space.
    inline void setPose(const hlsl::float64_t3& position, const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
    {
        m_gimbal.begin();
        m_gimbal.setPosition(position);
        m_gimbal.setOrientation(orientation);
        m_gimbal.end();
        m_gimbal.updateView();
    }

    /// @brief Replace only the tracked target position.
    inline void setPosition(const hlsl::float64_t3& position)
    {
        setPose(position, m_gimbal.getOrientation());
    }

    /// @brief Replace only the tracked target orientation.
    inline void setOrientation(const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
    {
        setPose(m_gimbal.getPosition(), orientation);
    }

    /// @brief Replace the tracked target pose from a rigid transform matrix when possible.
    inline bool trySetFromTransform(const hlsl::float64_t4x4& transform)
    {
        hlsl::float64_t3 position = hlsl::float64_t3(0.0);
        hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>();
        if (!hlsl::CCameraMathUtilities::tryExtractRigidPoseFromTransform(transform, position, orientation))
            return false;

        setPose(position, orientation);
        return true;
    }

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
    static inline hlsl::float64_t3 transformFollowLocalOffset(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& localOffset)
    {
        return hlsl::CCameraMathUtilities::rotateVectorByQuaternion(gimbal.getOrientation(), localOffset);
    }

    /// @brief Project a world-space offset into the tracked target local frame.
    static inline hlsl::float64_t3 projectFollowWorldOffsetToLocal(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& worldOffset)
    {
        return hlsl::CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame(gimbal.getOrientation(), worldOffset);
    }

    /// @brief Build a look-at orientation that points from `position` toward the tracked target.
    static inline bool buildFollowLookAtOrientation(
        const hlsl::float64_t3& position,
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& preferredUp,
        hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation)
    {
        return hlsl::CCameraMathUtilities::tryBuildLookAtOrientation(position, targetPosition, preferredUp, outOrientation);
    }

    /// @brief Capture world-space and target-local follow offsets from the current camera pose.
    static inline bool captureFollowOffsetsFromCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        SCameraFollowConfig& ioConfig)
    {
        const auto capture = solver.captureDetailed(camera);
        if (!capture.canUseGoal())
            return false;

        const auto& targetGimbal = trackedTarget.getGimbal();
        ioConfig.worldOffset = capture.goal.position - targetGimbal.getPosition();
        ioConfig.localOffset = projectFollowWorldOffsetToLocal(targetGimbal, ioConfig.worldOffset);
        return true;
    }

    /// @brief Measure the angular lock error between a camera forward axis and a tracked target.
    static inline bool tryComputeFollowTargetLockMetrics(
        const ICamera::CGimbal& cameraGimbal,
        const CTrackedTarget& trackedTarget,
        float& outAngleDeg,
        double* outDistance = nullptr)
    {
        const auto toTarget = trackedTarget.getGimbal().getPosition() - cameraGimbal.getPosition();
        const auto targetDistance = hlsl::length(toTarget);
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(targetDistance) || targetDistance <= SCameraToolingThresholds::TinyScalarEpsilon)
            return false;

        const auto forward = cameraGimbal.getZAxis();
        const auto forwardLength = hlsl::length(forward);
        if (!hlsl::CCameraMathUtilities::isFiniteVec3(forward) || !hlsl::CCameraMathUtilities::isFiniteScalar(forwardLength) || forwardLength <= SCameraToolingThresholds::TinyScalarEpsilon)
            return false;

        const auto forwardDirection = forward / forwardLength;
        const auto targetDir = toTarget / targetDistance;
        const auto dotForward = std::clamp(hlsl::dot(forwardDirection, targetDir), -1.0, 1.0);
        outAngleDeg = static_cast<float>(hlsl::degrees(hlsl::acos(dotForward)));
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(outAngleDeg))
            return false;

        if (outDistance)
            *outDistance = targetDistance;
        return true;
    }

    static inline bool tryBuildFollowPositionGoal(
        ICamera* camera,
        CCameraGoal& outGoal,
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const hlsl::float64_t3& preferredUp)
    {
        if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
            return CCameraGoalUtilities::buildCanonicalTargetRelativeGoalFromPosition(outGoal, targetPosition, position);

        outGoal.position = position;
        return buildFollowLookAtOrientation(outGoal.position, targetPosition, preferredUp, outGoal.orientation) && CCameraGoalUtilities::isGoalFinite(outGoal);
    }

    static inline bool tryBuildFollowGoal(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& config,
        CCameraGoal& outGoal)
    {
        if (!camera || !config.enabled || config.mode == ECameraFollowMode::Disabled)
            return false;

        const auto capture = solver.captureDetailed(camera);
        if (!capture.canUseGoal())
            return false;

        outGoal = capture.goal;

        const auto& targetGimbal = trackedTarget.getGimbal();
        const auto targetPosition = targetGimbal.getPosition();

        switch (config.mode)
        {
            case ECameraFollowMode::OrbitTarget:
            {
                if (!camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
                    return false;

                if (outGoal.hasPathState)
                {
                    return CCameraGoalUtilities::applyCanonicalPathGoalFields(outGoal, targetPosition, outGoal.pathState) && CCameraGoalUtilities::isGoalFinite(outGoal);
                }

                const bool hasSphericalState = outGoal.hasOrbitState || outGoal.hasDistance;
                if (!hasSphericalState)
                    return false;

                const auto orbitDistance = outGoal.hasOrbitState ? outGoal.orbitDistance : outGoal.distance;
                return CCameraGoalUtilities::applyCanonicalTargetRelativeGoal(
                    outGoal,
                    {
                        .target = targetPosition,
                        .orbitUv = outGoal.orbitUv,
                        .distance = orbitDistance
                    });
            }

            case ECameraFollowMode::LookAtTarget:
            {
                return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, capture.goal.position, targetGimbal.getYAxis());
            }

            case ECameraFollowMode::KeepWorldOffset:
            {
                const auto position = targetPosition + config.worldOffset;
                return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, position, targetGimbal.getYAxis());
            }

            case ECameraFollowMode::KeepLocalOffset:
            {
                const auto position = targetPosition + transformFollowLocalOffset(targetGimbal, config.localOffset);
                return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, position, targetGimbal.getYAxis());
            }

            default:
                return false;
        }
    }

    static inline CCameraGoalSolver::SApplyResult applyFollowToCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& config,
        CCameraGoal* outGoal = nullptr)
    {
        CCameraGoal goal = {};
        if (!tryBuildFollowGoal(solver, camera, trackedTarget, config, goal))
            return {};

        if (outGoal)
            *outGoal = goal;

        return solver.applyDetailed(camera, goal);
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_FOLLOW_UTILITIES_HPP_

