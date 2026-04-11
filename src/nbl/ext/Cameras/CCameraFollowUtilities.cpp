// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraFollowUtilities.hpp"

namespace nbl::core
{

CTrackedTarget::CTrackedTarget(
    const hlsl::float64_t3& position,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation,
    std::string identifier)
    : m_identifier(std::move(identifier)),
    m_gimbal(gimbal_t::base_t::SCreationParameters{ .position = position, .orientation = orientation })
{
    m_gimbal.updateView();
}

void CTrackedTarget::setPose(const hlsl::float64_t3& position, const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
{
    m_gimbal.begin();
    m_gimbal.setPosition(position);
    m_gimbal.setOrientation(orientation);
    m_gimbal.end();
    m_gimbal.updateView();
}

void CTrackedTarget::setPosition(const hlsl::float64_t3& position)
{
    setPose(position, m_gimbal.getOrientation());
}

void CTrackedTarget::setOrientation(const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
{
    setPose(m_gimbal.getPosition(), orientation);
}

bool CTrackedTarget::trySetFromTransform(const hlsl::float64_t4x4& transform)
{
    hlsl::float64_t3 position = hlsl::float64_t3(0.0);
    hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>();
    if (!hlsl::CCameraMathUtilities::tryExtractRigidPoseFromTransform(transform, position, orientation))
        return false;

    setPose(position, orientation);
    return true;
}

hlsl::float64_t3 CCameraFollowUtilities::transformFollowLocalOffset(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& localOffset)
{
    return hlsl::CCameraMathUtilities::rotateVectorByQuaternion(gimbal.getOrientation(), localOffset);
}

hlsl::float64_t3 CCameraFollowUtilities::projectFollowWorldOffsetToLocal(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& worldOffset)
{
    return hlsl::CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame(gimbal.getOrientation(), worldOffset);
}

bool CCameraFollowUtilities::buildFollowLookAtOrientation(
    const hlsl::float64_t3& position,
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& preferredUp,
    hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation)
{
    return hlsl::CCameraMathUtilities::tryBuildLookAtOrientation(position, targetPosition, preferredUp, outOrientation);
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
    ioConfig.worldOffset = capture.goal.position - targetGimbal.getPosition();
    ioConfig.localOffset = projectFollowWorldOffsetToLocal(targetGimbal, ioConfig.worldOffset);
    return true;
}

bool CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(
    const ICamera::CGimbal& cameraGimbal,
    const CTrackedTarget& trackedTarget,
    float& outAngleDeg,
    double* outDistance)
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
    return buildFollowLookAtOrientation(outGoal.position, targetPosition, preferredUp, outGoal.orientation) && CCameraGoalUtilities::isGoalFinite(outGoal);
}

bool CCameraFollowUtilities::tryBuildFollowGoal(
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

} // namespace nbl::core
