// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraFollowRegressionUtilities.hpp"

namespace nbl::system
{

SCameraFollowRegressionThresholds CCameraFollowRegressionUtilities::makeFollowRegressionThresholds(
    const float projectedNdcTolerance,
    const float lockAngleToleranceDeg)
{
    auto thresholds = SCameraFollowRegressionThresholds{};
    thresholds.projectedNdcTolerance = projectedNdcTolerance;
    thresholds.lockAngleToleranceDeg = lockAngleToleranceDeg;
    return thresholds;
}

bool CCameraFollowRegressionUtilities::tryComputeProjectedFollowTargetMetrics(
    const SCameraProjectionContext& projectionContext,
    const core::CTrackedTarget& trackedTarget,
    SCameraProjectedTargetMetrics& outMetrics,
    const float clipWEpsilon)
{
    outMetrics = {};
    const hlsl::float32_t3 target = hlsl::CCameraMathUtilities::castVector<hlsl::float32_t>(trackedTarget.getGimbal().getPosition());
    const auto viewSpace = hlsl::mul(projectionContext.viewMatrix, hlsl::float32_t4(target.x, target.y, target.z, 1.0f));
    const auto clipProjection = hlsl::transpose(projectionContext.projectionMatrix);
    const auto clip = hlsl::mul(clipProjection, viewSpace);
    if (!hlsl::CCameraMathUtilities::isFiniteScalar(clip.x) || !hlsl::CCameraMathUtilities::isFiniteScalar(clip.y) || !hlsl::CCameraMathUtilities::isFiniteScalar(clip.z) || !hlsl::CCameraMathUtilities::isFiniteScalar(clip.w))
        return false;

    const auto absW = hlsl::abs(clip.w);
    if (absW < clipWEpsilon)
        return false;

    const float invW = 1.0f / clip.w;
    outMetrics.ndc = hlsl::float32_t2(clip.x, clip.y) * invW;
    if (!hlsl::CCameraMathUtilities::isFiniteScalar(outMetrics.ndc.x) || !hlsl::CCameraMathUtilities::isFiniteScalar(outMetrics.ndc.y))
        return false;

    outMetrics.radius = hlsl::length(outMetrics.ndc);
    return true;
}

bool CCameraFollowRegressionUtilities::validateProjectedFollowTargetContract(
    const SCameraProjectionContext& projectionContext,
    const core::CTrackedTarget& trackedTarget,
    SCameraProjectedTargetMetrics& outMetrics,
    std::string* error,
    const SCameraFollowRegressionThresholds& thresholds)
{
    if (!tryComputeProjectedFollowTargetMetrics(projectionContext, trackedTarget, outMetrics, thresholds.clipWEpsilon))
    {
        if (error)
            *error = "failed to project follow target";
        return false;
    }

    if (outMetrics.radius > thresholds.projectedNdcTolerance)
    {
        if (error)
        {
            *error = "projected target mismatch ndc=(" + std::to_string(outMetrics.ndc.x) +
                "," + std::to_string(outMetrics.ndc.y) + ") radius=" + std::to_string(outMetrics.radius);
        }
        return false;
    }

    return true;
}

SCameraFollowVisualMetrics CCameraFollowRegressionUtilities::buildFollowVisualMetrics(
    core::ICamera* camera,
    const core::CTrackedTarget& trackedTarget,
    const core::SCameraFollowConfig* followConfig,
    const SCameraProjectionContext* projectionContext)
{
    SCameraFollowVisualMetrics out = {};
    if (!camera || !followConfig || !followConfig->enabled || followConfig->mode == core::ECameraFollowMode::Disabled)
        return out;

    out.active = true;
    out.mode = followConfig->mode;

    double targetDistance = 0.0;
    out.lockValid = core::CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(followConfig->mode) &&
        core::CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &targetDistance);
    if (out.lockValid)
        out.targetDistance = static_cast<float>(targetDistance);

    if (out.lockValid && projectionContext)
        out.projectedValid = tryComputeProjectedFollowTargetMetrics(*projectionContext, trackedTarget, out.projectedTarget);

    return out;
}

bool CCameraFollowRegressionUtilities::validateFollowTargetContract(
    core::ICamera* camera,
    const core::CTrackedTarget& trackedTarget,
    const core::SCameraFollowConfig& followConfig,
    const core::CCameraGoal& followGoal,
    SCameraFollowRegressionResult& out,
    std::string* error,
    const SCameraProjectionContext* projectionContext,
    const SCameraFollowRegressionThresholds& thresholds)
{
    out = {};
    if (!camera)
    {
        if (error)
            *error = "missing camera";
        return false;
    }

    if (core::CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(followConfig.mode))
    {
        out.hasLockMetrics = core::CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &out.targetDistance);
        if (!out.hasLockMetrics)
        {
            if (error)
                *error = "failed to compute follow lock metrics";
            return false;
        }

        const auto& trackedTargetGimbal = trackedTarget.getGimbal();
        const auto& cameraGimbal = camera->getGimbal();
        const hlsl::float64_t3 trackedTargetPosition = trackedTargetGimbal.getPosition();
        const hlsl::float64_t3 cameraPosition = cameraGimbal.getPosition();
        const double expectedTargetDistance = hlsl::length(trackedTargetPosition - cameraPosition);
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(expectedTargetDistance) || hlsl::abs(expectedTargetDistance - out.targetDistance) > thresholds.distanceTolerance)
        {
            if (error)
            {
                *error = "target distance mismatch actual=" + std::to_string(out.targetDistance) +
                    " expected=" + std::to_string(expectedTargetDistance);
            }
            return false;
        }

        if (out.lockAngleDeg > thresholds.lockAngleToleranceDeg)
        {
            if (error)
                *error = "lock angle mismatch angle_deg=" + std::to_string(out.lockAngleDeg);
            return false;
        }

        if (projectionContext)
        {
            out.hasProjectedMetrics = tryComputeProjectedFollowTargetMetrics(
                *projectionContext,
                trackedTarget,
                out.projectedTarget,
                thresholds.clipWEpsilon);
            if (!out.hasProjectedMetrics)
            {
                if (error)
                    *error = "failed to compute projected follow target metrics";
                return false;
            }

            if (out.projectedTarget.radius > thresholds.projectedNdcTolerance)
            {
                if (error)
                {
                    *error = "projected target mismatch ndc=(" + std::to_string(out.projectedTarget.ndc.x) +
                        "," + std::to_string(out.projectedTarget.ndc.y) + ") radius=" + std::to_string(out.projectedTarget.radius);
                }
                return false;
            }
        }
    }

    if (camera->supportsGoalState(core::ICamera::GoalStateSphericalTarget))
    {
        core::ICamera::SphericalTargetState state;
        if (!camera->tryGetSphericalTargetState(state))
        {
            if (error)
                *error = "missing spherical target state";
            return false;
        }

        out.hasSphericalState = true;
        out.sphericalTarget = state.target;
        out.sphericalDistance = state.distance;

        const auto& trackedTargetGimbal = trackedTarget.getGimbal();
        const auto& cameraGimbal = camera->getGimbal();
        const hlsl::float64_t3 trackedTargetPosition = trackedTargetGimbal.getPosition();
        const hlsl::float64_t3 targetDelta = state.target - trackedTargetPosition;
        const double targetDeltaLen = hlsl::length(targetDelta);
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(targetDeltaLen) || targetDeltaLen > thresholds.targetTolerance)
        {
            if (error)
                *error = "spherical target writeback mismatch";
            return false;
        }

        const double actualDistance = hlsl::length(cameraGimbal.getPosition() - trackedTargetPosition);
        const auto expectedDistance = followGoal.hasOrbitState ? static_cast<double>(followGoal.orbitDistance) :
            (followGoal.hasDistance ? static_cast<double>(followGoal.distance) : actualDistance);
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(actualDistance) || !hlsl::CCameraMathUtilities::isFiniteScalar(expectedDistance) ||
            hlsl::abs(actualDistance - expectedDistance) > thresholds.distanceTolerance ||
            hlsl::abs(static_cast<double>(state.distance) - expectedDistance) > thresholds.distanceTolerance)
        {
            if (error)
            {
                *error = "spherical distance mismatch actual=" + std::to_string(actualDistance) +
                    " state=" + std::to_string(state.distance) +
                    " expected=" + std::to_string(expectedDistance);
            }
            return false;
        }
    }

    out.passed = true;
    return true;
}

bool CCameraFollowRegressionUtilities::buildApplyAndValidateFollowTargetContract(
    const core::CCameraGoalSolver& solver,
    core::ICamera* camera,
    const core::CTrackedTarget& trackedTarget,
    const core::SCameraFollowConfig& followConfig,
    SCameraFollowApplyValidationResult& out,
    std::string* error,
    const SCameraProjectionContext* projectionContext,
    const SCameraFollowRegressionThresholds& thresholds)
{
    out = {};

    if (!core::CCameraFollowUtilities::tryBuildFollowGoal(solver, camera, trackedTarget, followConfig, out.goal))
    {
        if (error)
            *error = "failed to build follow goal";
        return false;
    }
    out.hasGoal = true;

    out.applyResult = core::CCameraFollowUtilities::applyFollowToCamera(solver, camera, trackedTarget, followConfig);
    if (!out.applyResult.succeeded())
    {
        if (error)
            *error = "failed to apply follow goal";
        return false;
    }

    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
    {
        if (error)
            *error = "failed to capture camera state after follow apply";
        return false;
    }

    out.hasCapturedGoal = true;
    out.capturedGoal = capture.goal;
    if (!core::CCameraGoalUtilities::compareGoals(out.capturedGoal, out.goal, thresholds.positionTolerance, thresholds.rotationToleranceDeg, thresholds.scalarTolerance))
    {
        if (error)
            *error = std::string("follow goal mismatch. ") + core::CCameraGoalUtilities::describeGoalMismatch(out.capturedGoal, out.goal);
        return false;
    }

    return validateFollowTargetContract(
        camera,
        trackedTarget,
        followConfig,
        out.goal,
        out.regression,
        error,
        projectionContext,
        thresholds);
}

} // namespace nbl::system
