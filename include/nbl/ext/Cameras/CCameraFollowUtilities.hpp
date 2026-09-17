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

namespace nbl::ext::cameras
{

/// @brief Reusable tracked-target and follow helpers.
///
/// The tracked subject owns its own gimbal. Follow code reads that pose and
/// maps one camera plus one tracked target into a `CCameraGoal`.
class CTrackedTarget
{
public:
    using gimbal_t = IGimbal;

    /// @brief Construct a tracked target from an initial pose and optional identifier.
    CTrackedTarget(
        const hlsl::float64_t3& position = hlsl::float64_t3(0.0),
        const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity(),
        std::string identifier = "Follow Target");

    /// @brief Return the stable human-readable identifier of the tracked target.
    inline const std::string& getIdentifier() const { return m_identifier; }
    /// @brief Return read-only access to the tracked target gimbal.
    inline const gimbal_t& getGimbal() const { return m_gimbal; }
    /// @brief Return mutable access to the tracked target gimbal.
    inline gimbal_t& getGimbal() { return m_gimbal; }

    /// @brief Replace the tracked target pose in world space.
    void setPose(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation);

    /// @brief Replace only the tracked target position.
    void setPosition(const hlsl::float64_t3& position);

    /// @brief Replace only the tracked target orientation.
    void setOrientation(const hlsl::math::quaternion<hlsl::float64_t>& orientation);

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
/// - `KeepWorldOffset` places the camera at `trackedTarget.position + offset` and looks at the target
/// - `KeepLocalOffset` transforms `offset` by the tracked-target local frame and looks at the target
///
/// The tracked target provides pose data. The camera reads that data and does
/// not own the tracked subject.
enum class ECameraFollowMode : uint8_t
{
    Unknown,
    OrbitTarget,
    LookAtTarget,
    KeepWorldOffset,
    KeepLocalOffset
};

/// @brief Reusable follow configuration interpreted against a tracked target gimbal.
struct SCameraFollowConfig
{
    /// @brief Whether follow should be applied at all.
    bool enabled = false;
    /// @brief Follow policy used when the configuration is enabled.
    ECameraFollowMode mode = ECameraFollowMode::OrbitTarget;
    /// @brief Camera-to-target offset in the frame the mode reads it in: world space under `KeepWorldOffset`,
    /// tracked-target local space under `KeepLocalOffset`, unused by the other modes.
    /// `captureFollowOffsetsFromCamera` writes it in whichever frame the current mode needs.
    hlsl::float64_t3 offset = hlsl::float64_t3(0.0);
};

/// @brief Shared policy helpers for tracked-target follow.
///
/// The helpers decide which follow modes lock the view, how offsets are captured,
/// and how a tracked target is translated into a `CCameraGoal` that can then be
/// applied through the shared goal solver.
struct CCameraFollowUtilities final
{
    /// @brief Return whether the follow mode rebuilds camera orientation toward the tracked target.
    static bool cameraFollowModeLocksViewToTarget(ECameraFollowMode mode);

    /// @brief Return whether the follow mode reads `SCameraFollowConfig::offset`, which has to be captured first.
    static bool cameraFollowModeUsesCapturedOffset(ECameraFollowMode mode);

    /// @brief Build the shared default follow configuration for one camera instance; a null camera gives a disabled one.
    static SCameraFollowConfig makeDefaultFollowConfig(const ICamera* camera);

    /// @brief Store the current camera-to-target offset into `ioConfig`, in the frame `ioConfig.mode` reads it in.
    static bool captureFollowOffsetsFromCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        SCameraFollowConfig& ioConfig);

    /// @brief Measure the angular lock error between a camera forward axis and a tracked target.
    /// @param outDistance optional (may be null); receives the camera-to-target distance on success.
    static bool tryComputeFollowTargetLockMetrics(
        const IGimbal& cameraGimbal,
        const CTrackedTarget& trackedTarget,
        hlsl::float64_t& outAngleDeg,
        hlsl::float64_t* outDistance = nullptr);

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

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_FOLLOW_UTILITIES_HPP_

