// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraFollowUtilities.hpp"

namespace nbl::ext::cameras
{

CTrackedTarget::CTrackedTarget(
    const hlsl::float64_t3& position,
    const hlsl::math::quaternion<hlsl::float64_t>& orientation,
    std::string identifier)
    : m_identifier(std::move(identifier)),
    m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation })
{
}

void CTrackedTarget::setPose(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation)
{
    m_gimbal.setPose(SCameraRigPose{ .position = position, .orientation = orientation });
}

void CTrackedTarget::setPosition(const hlsl::float64_t3& position)
{
    setPose(position, m_gimbal.getOrientation());
}

void CTrackedTarget::setOrientation(const hlsl::math::quaternion<hlsl::float64_t>& orientation)
{
    setPose(m_gimbal.getPosition(), orientation);
}

bool CTrackedTarget::trySetFromTransform(const hlsl::float64_t4x4& transform)
{
    return m_gimbal.setPose(transform);
}

bool CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(const ECameraFollowMode mode)
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

bool CCameraFollowUtilities::cameraFollowModeUsesCapturedOffset(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepWorldOffset || mode == ECameraFollowMode::KeepLocalOffset;
}

SCameraFollowConfig CCameraFollowUtilities::makeDefaultFollowConfig(const ICamera* const camera)
{
    if (!camera)
        return {};

    auto mode = ECameraFollowMode::Unknown;
    switch (camera->getKind())
    {
        case ICamera::CameraKind::Orbit:
        case ICamera::CameraKind::Arcball:
        case ICamera::CameraKind::Turntable:
        case ICamera::CameraKind::TopDown:
        case ICamera::CameraKind::Isometric:
        case ICamera::CameraKind::DollyZoom:
        case ICamera::CameraKind::Path:
            mode = ECameraFollowMode::OrbitTarget;
            break;
        case ICamera::CameraKind::Chase:
        case ICamera::CameraKind::Dolly:
            mode = ECameraFollowMode::KeepLocalOffset;
            break;
        default:
            break;
    }

    return {
        .enabled = mode != ECameraFollowMode::Unknown,
        .mode = mode
    };
}

bool CCameraFollowUtilities::captureFollowOffsetsFromCamera(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    SCameraFollowConfig& ioConfig)
{
    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
        return false;

    const auto& targetGimbal = trackedTarget.getGimbal();
    const auto worldOffset = capture.goal.position - targetGimbal.getPosition();

    // `KeepLocalOffset` replays the offset through the target's orientation, so it is stored in the target's
    // frame: rotating the world offset by the inverse of that orientation.
    ioConfig.offset = (ioConfig.mode == ECameraFollowMode::KeepLocalOffset)
        ? CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame(targetGimbal.getOrientation(), worldOffset)
        : worldOffset;
    return true;
}

bool CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(
    const IGimbal& cameraGimbal,
    const CTrackedTarget& trackedTarget,
    hlsl::float64_t& outAngleDeg,
    hlsl::float64_t* outDistance)
{
    const auto toTarget = trackedTarget.getGimbal().getPosition() - cameraGimbal.getPosition();
    const auto targetDistance = hlsl::length(toTarget);
    if (!CCameraMathUtilities::isFiniteScalar(targetDistance) || targetDistance <= SCameraToolingThresholds::TinyScalarEpsilon)
        return false;

    const auto forward = cameraGimbal.getForward();
    const auto forwardLength = hlsl::length(forward);
    if (!CCameraMathUtilities::isFiniteVec3(forward) || !CCameraMathUtilities::isFiniteScalar(forwardLength) || forwardLength <= SCameraToolingThresholds::TinyScalarEpsilon)
        return false;

    const auto forwardDirection = forward / forwardLength;
    const auto targetDir = toTarget / targetDistance;
    const auto dotForward = std::clamp(hlsl::dot(forwardDirection, targetDir), -1.0, 1.0);
    outAngleDeg = hlsl::degrees(hlsl::acos(dotForward));
    if (!CCameraMathUtilities::isFiniteScalar(outAngleDeg))
        return false;

    if (outDistance)
        *outDistance = targetDistance;
    return true;
}

bool CCameraFollowUtilities::tryBuildFollowPositionGoal(
    ICamera* camera,
    CCameraGoal& outGoal,
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const hlsl::float64_t3& preferredUp)
{
    if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
        return CCameraGoalUtilities::buildCanonicalTargetRelativeGoalFromPosition(outGoal, targetPosition, position);

    outGoal.position = position;
    return CCameraMathUtilities::tryBuildLookAtOrientation(outGoal.position, targetPosition, preferredUp, outGoal.orientation) &&
        CCameraGoalUtilities::isGoalFinite(outGoal);
}

bool CCameraFollowUtilities::tryBuildFollowGoal(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& config,
    CCameraGoal& outGoal)
{
    if (!camera || !config.enabled || config.mode == ECameraFollowMode::Unknown)
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
                    .angles = outGoal.orbitUv,
                    .distance = orbitDistance
                });
        }

        case ECameraFollowMode::LookAtTarget:
        {
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, capture.goal.position, targetGimbal.getUp());
        }

        case ECameraFollowMode::KeepWorldOffset:
        {
            const auto position = targetPosition + config.offset;
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, position, targetGimbal.getUp());
        }

        case ECameraFollowMode::KeepLocalOffset:
        {
            // the offset is stored in the target's frame, so it rotates with the target before it is applied
            const auto worldOffset = targetGimbal.getOrientation().transformVector(config.offset, true);
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, targetPosition + worldOffset, targetGimbal.getUp());
        }

        default:
            return false;
    }
}

CCameraGoalSolver::SApplyResult CCameraFollowUtilities::applyFollowToCamera(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& config,
    CCameraGoal* outGoal)
{
    CCameraGoal goal = {};
    if (!tryBuildFollowGoal(solver, camera, trackedTarget, config, goal))
        return {};

    if (outGoal)
        *outGoal = goal;

    return solver.applyDetailed(camera, goal);
}

} // namespace nbl::ext::cameras
