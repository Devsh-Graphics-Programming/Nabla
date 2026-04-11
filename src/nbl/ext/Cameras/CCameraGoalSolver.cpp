// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraGoalSolver.hpp"

#include <limits>

namespace nbl::core
{

bool CCameraGoalSolver::buildEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
{
    out.clear();
    if (!camera)
        return false;

    const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);

    if (camera->hasCapability(ICamera::SphericalTarget))
        return buildSphericalEvents(camera, canonicalTarget, out);

    return buildFreeEvents(camera, canonicalTarget, out);
}

bool CCameraGoalSolver::capture(ICamera* camera, CCameraGoal& out) const
{
    out = {};
    if (!camera)
        return false;

    const ICamera::CGimbal& gimbal = camera->getGimbal();
    out.position = hlsl::float64_t3(gimbal.getPosition());
    out.orientation = gimbal.getOrientation();
    out.sourceKind = camera->getKind();
    out.sourceCapabilities = ICamera::capability_flags_t(camera->getCapabilities());
    out.sourceGoalStateMask = ICamera::goal_state_flags_t(camera->getGoalStateMask());

    ICamera::SphericalTargetState sphericalState;
    if (camera->tryGetSphericalTargetState(sphericalState))
    {
        out.targetPosition = sphericalState.target;
        out.hasTargetPosition = true;
        out.distance = sphericalState.distance;
        out.hasDistance = true;
        out.orbitDistance = sphericalState.distance;
        out.orbitUv = sphericalState.orbitUv;
        out.hasOrbitState = true;
    }

    ICamera::DynamicPerspectiveState dynamicState;
    if (camera->tryGetDynamicPerspectiveState(dynamicState))
    {
        out.hasDynamicPerspectiveState = true;
        out.dynamicPerspectiveState = dynamicState;
    }

    ICamera::PathState pathState;
    if (camera->tryGetPathState(pathState))
    {
        out.hasPathState = true;
        out.pathState = pathState;
    }

    out = CCameraGoalUtilities::canonicalizeGoal(out);
    return true;
}

CCameraGoalSolver::SCaptureResult CCameraGoalSolver::captureDetailed(ICamera* camera) const
{
    SCaptureResult result;
    result.hasCamera = camera != nullptr;
    if (!result.hasCamera)
        return result;

    result.captured = capture(camera, result.goal);
    result.finiteGoal = result.captured && CCameraGoalUtilities::isGoalFinite(result.goal);
    return result;
}

CCameraGoalSolver::SCompatibilityResult CCameraGoalSolver::analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const
{
    SCompatibilityResult result;
    if (!camera)
        return result;

    const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);
    result.sameKind = canonicalTarget.sourceKind == ICamera::CameraKind::Unknown || canonicalTarget.sourceKind == camera->getKind();
    result.supportedGoalStateMask = ICamera::goal_state_flags_t(camera->getGoalStateMask());
    result.requiredGoalStateMask = CCameraGoalUtilities::getRequiredGoalStateMask(canonicalTarget);
    result.missingGoalStateMask = result.requiredGoalStateMask & ~result.supportedGoalStateMask;
    result.exact = result.missingGoalStateMask == ICamera::GoalStateNone;
    return result;
}

CCameraGoalSolver::SApplyResult CCameraGoalSolver::applyDetailed(ICamera* camera, const CCameraGoal& target) const
{
    SApplyResult result;
    if (!camera)
        return result;

    const auto canonicalTarget = CCameraGoalUtilities::canonicalizeGoal(target);

    bool exact = true;
    bool absoluteChanged = false;

    if (!camera->hasCapability(ICamera::SphericalTarget))
    {
        bool poseChanged = false;
        bool poseExact = false;
        if (tryApplyAbsoluteReferencePose(camera, canonicalTarget, poseChanged, poseExact))
        {
            result.issues |= SApplyResult::EIssue::UsedAbsolutePoseFallback;
            absoluteChanged = absoluteChanged || poseChanged;
            if (poseExact && !canonicalTarget.hasDynamicPerspectiveState)
            {
                result.status = poseChanged ?
                    SApplyResult::EStatus::AppliedAbsoluteOnly :
                    SApplyResult::EStatus::AlreadySatisfied;
                result.exact = true;
                return result;
            }
        }
    }

    if (canonicalTarget.hasTargetPosition)
    {
        ICamera::SphericalTargetState beforeState;
        if (!camera->tryGetSphericalTargetState(beforeState))
        {
            result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
            exact = false;
        }
        else
        {
            const auto beforeTarget = beforeState.target;
            if (!camera->trySetSphericalTarget(canonicalTarget.targetPosition))
            {
                result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                ICamera::SphericalTargetState afterState;
                if (!camera->tryGetSphericalTargetState(afterState))
                {
                    result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                    exact = false;
                }
                else
                {
                    absoluteChanged = afterState.target != beforeTarget;
                    exact = exact && afterState.target == canonicalTarget.targetPosition;
                }
            }
        }
    }

    if (canonicalTarget.hasDistance || canonicalTarget.hasOrbitState)
    {
        ICamera::SphericalTargetState beforeState;
        if (!camera->tryGetSphericalTargetState(beforeState))
        {
            result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
            exact = false;
        }
        else
        {
            const float desiredDistance = canonicalTarget.hasOrbitState ? canonicalTarget.orbitDistance : canonicalTarget.distance;
            const float beforeDistance = beforeState.distance;
            if (!camera->trySetSphericalDistance(desiredDistance))
            {
                result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                exact = false;
            }
            else
            {
                ICamera::SphericalTargetState afterState;
                if (!camera->tryGetSphericalTargetState(afterState))
                {
                    result.issues |= SApplyResult::EIssue::MissingSphericalTargetState;
                    exact = false;
                }
                else
                {
                    absoluteChanged = absoluteChanged || afterState.distance != beforeDistance;
                    exact = exact && hlsl::abs(static_cast<double>(afterState.distance - desiredDistance)) <= SCameraToolingThresholds::ScalarTolerance;
                }
            }
        }
    }

    if (canonicalTarget.hasPathState)
    {
        ICamera::PathState beforeState;
        if (!camera->tryGetPathState(beforeState))
        {
            result.issues |= SApplyResult::EIssue::MissingPathState;
            exact = false;
        }
        else if (!camera->trySetPathState(canonicalTarget.pathState))
        {
            result.issues |= SApplyResult::EIssue::MissingPathState;
            exact = false;
        }
        else
        {
            ICamera::PathState afterState;
            if (!camera->tryGetPathState(afterState))
            {
                result.issues |= SApplyResult::EIssue::MissingPathState;
                exact = false;
            }
            else
            {
                const auto thresholds = SCameraPathDefaults::ComparisonThresholds;
                const bool pathChanged = CCameraPathUtilities::pathStatesChanged(beforeState, afterState, thresholds);
                const bool pathExact = CCameraPathUtilities::pathStatesNearlyEqual(afterState, canonicalTarget.pathState, thresholds);

                absoluteChanged = absoluteChanged || pathChanged;
                exact = exact && pathExact;
            }
        }
    }

    if (canonicalTarget.hasDynamicPerspectiveState)
    {
        ICamera::DynamicPerspectiveState beforeState;
        if (!camera->tryGetDynamicPerspectiveState(beforeState))
        {
            result.issues |= SApplyResult::EIssue::MissingDynamicPerspectiveState;
            exact = false;
        }
        else if (!camera->trySetDynamicPerspectiveState(canonicalTarget.dynamicPerspectiveState))
        {
            result.issues |= SApplyResult::EIssue::MissingDynamicPerspectiveState;
            exact = false;
        }
        else
        {
            ICamera::DynamicPerspectiveState afterState;
            if (!camera->tryGetDynamicPerspectiveState(afterState))
            {
                result.issues |= SApplyResult::EIssue::MissingDynamicPerspectiveState;
                exact = false;
            }
            else
            {
                const bool dynamicChanged = !hlsl::CCameraMathUtilities::nearlyEqualScalar(beforeState.baseFov, afterState.baseFov, static_cast<float>(SCameraToolingThresholds::ScalarTolerance)) ||
                    !hlsl::CCameraMathUtilities::nearlyEqualScalar(beforeState.referenceDistance, afterState.referenceDistance, static_cast<float>(SCameraToolingThresholds::ScalarTolerance));
                const bool dynamicExact = hlsl::CCameraMathUtilities::nearlyEqualScalar(afterState.baseFov, canonicalTarget.dynamicPerspectiveState.baseFov, static_cast<float>(SCameraToolingThresholds::ScalarTolerance)) &&
                    hlsl::CCameraMathUtilities::nearlyEqualScalar(afterState.referenceDistance, canonicalTarget.dynamicPerspectiveState.referenceDistance, static_cast<float>(SCameraToolingThresholds::ScalarTolerance));

                absoluteChanged = absoluteChanged || dynamicChanged;
                exact = exact && dynamicExact;
            }
        }
    }

    std::vector<CVirtualGimbalEvent> events;
    buildEvents(camera, canonicalTarget, events);
    result.eventCount = static_cast<uint32_t>(events.size());
    result.exact = exact;

    if (events.empty())
    {
        if (absoluteChanged)
            result.status = SApplyResult::EStatus::AppliedAbsoluteOnly;
        else if (exact)
            result.status = SApplyResult::EStatus::AlreadySatisfied;
        return result;
    }

    if (camera->manipulate({ events.data(), events.size() }))
    {
        result.status = absoluteChanged ?
            SApplyResult::EStatus::AppliedAbsoluteAndVirtualEvents :
            SApplyResult::EStatus::AppliedVirtualEvents;
        return result;
    }

    if (absoluteChanged)
    {
        result.status = SApplyResult::EStatus::AppliedAbsoluteOnly;
        result.exact = false;
        return result;
    }

    result.issues |= SApplyResult::EIssue::VirtualEventReplayFailed;
    result.status = SApplyResult::EStatus::Failed;
    result.exact = false;
    return result;
}

bool CCameraGoalSolver::apply(ICamera* camera, const CCameraGoal& target) const
{
    return applyDetailed(camera, target).succeeded();
}

void CCameraGoalSolver::appendYawPitchRollEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const hlsl::float64_t3& eulerRadians,
    double denominator,
    bool includeRoll) const
{
    static constexpr std::array<SCameraVirtualEventAxisBinding, 3u> RotationBindings = {{
        { CVirtualGimbalEvent::TiltUp, CVirtualGimbalEvent::TiltDown },
        { CVirtualGimbalEvent::PanRight, CVirtualGimbalEvent::PanLeft },
        { CVirtualGimbalEvent::RollRight, CVirtualGimbalEvent::RollLeft }
    }};

    auto tolerances = SGoalSolverDefaults::AngularToleranceDegVec;
    if (!includeRoll)
        tolerances.z = std::numeric_limits<hlsl::float64_t>::infinity();

    CCameraVirtualEventUtilities::appendAngularAxisEvents(
        events,
        eulerRadians,
        hlsl::float64_t3(denominator),
        tolerances,
        RotationBindings);
}

void CCameraGoalSolver::appendPathDeltaEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const SCameraPathDelta& delta,
    double moveDenominator,
    double rotationDenominator) const
{
    CCameraPathUtilities::appendPathDeltaEvents(
        events,
        delta,
        moveDenominator,
        rotationDenominator,
        SCameraPathDefaults::ExactComparisonThresholds);
}

double CCameraGoalSolver::getMoveMagnitudeDenominator(const ICamera* camera) const
{
    const double moveScale = camera->getMoveSpeedScale();
    return camera->getUnscaledVirtualTranslationMagnitude() * (moveScale == 0.0 ? SGoalSolverDefaults::UnitScale : moveScale);
}

double CCameraGoalSolver::getRotationMagnitudeDenominator(const ICamera* camera) const
{
    const double rotationScale = camera->getRotationSpeedScale();
    return rotationScale == 0.0 ? SGoalSolverDefaults::UnitScale : rotationScale;
}

bool CCameraGoalSolver::computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const
{
    outPositionDelta = 0.0;
    outRotationDeltaDeg = 0.0;
    if (!camera)
        return false;

    const ICamera::CGimbal& gimbal = camera->getGimbal();
    hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
    if (!hlsl::CCameraMathUtilities::tryComputePoseDelta<hlsl::float64_t>(gimbal.getPosition(), gimbal.getOrientation(), target.position, target.orientation, poseDelta))
        return false;

    outPositionDelta = poseDelta.position;
    outRotationDeltaDeg = poseDelta.rotationDeg;
    return true;
}

bool CCameraGoalSolver::tryApplyAbsoluteReferencePose(ICamera* camera, const CCameraGoal& target, bool& outChanged, bool& outExact) const
{
    outChanged = false;
    outExact = false;
    if (!camera)
        return false;

    switch (camera->getKind())
    {
        case ICamera::CameraKind::Free:
        case ICamera::CameraKind::FPS:
            break;
        default:
            return false;
    }

    double beforePosDelta = 0.0;
    double beforeRotDeltaDeg = 0.0;
    if (!computePoseMismatch(camera, target, beforePosDelta, beforeRotDeltaDeg))
        return false;

    if (beforePosDelta <= SCameraToolingThresholds::DefaultPositionTolerance && beforeRotDeltaDeg <= SCameraToolingThresholds::DefaultAngularToleranceDeg)
    {
        outExact = true;
        return true;
    }

    const auto targetFrame = hlsl::CCameraMathUtilities::composeTransformMatrix(target.position, target.orientation);

    camera->manipulate({}, &targetFrame);

    double afterPosDelta = 0.0;
    double afterRotDeltaDeg = 0.0;
    if (!computePoseMismatch(camera, target, afterPosDelta, afterRotDeltaDeg))
        return false;

    outChanged = !hlsl::CCameraMathUtilities::isNearlyZeroScalar(afterPosDelta - beforePosDelta, static_cast<double>(SCameraToolingThresholds::TinyScalarEpsilon)) ||
        !hlsl::CCameraMathUtilities::isNearlyZeroScalar(afterRotDeltaDeg - beforeRotDeltaDeg, static_cast<double>(SCameraToolingThresholds::TinyScalarEpsilon));
    outExact = afterPosDelta <= SCameraToolingThresholds::DefaultPositionTolerance && afterRotDeltaDeg <= SCameraToolingThresholds::DefaultAngularToleranceDeg;
    return true;
}

bool CCameraGoalSolver::buildTargetRelativeEvents(
    ICamera* camera,
    const ICamera::SphericalTargetState& sphericalState,
    const SCameraTargetRelativeState& goal,
    std::vector<CVirtualGimbalEvent>& out,
    const SCameraTargetRelativeEventPolicy& policy) const
{
    const auto delta = CCameraTargetRelativeUtilities::buildTargetRelativeDelta(sphericalState, goal);
    CCameraTargetRelativeUtilities::appendTargetRelativeDeltaEvents(
        out,
        delta,
        policy.translateOrbit ? getMoveMagnitudeDenominator(camera) : getRotationMagnitudeDenominator(camera),
        SCameraToolingThresholds::DefaultAngularToleranceDeg,
        camera->getUnscaledVirtualTranslationMagnitude(),
        SCameraToolingThresholds::ScalarTolerance,
        policy);
    return !out.empty();
}

bool CCameraGoalSolver::buildPathEvents(
    ICamera* camera,
    const CCameraGoal& target,
    const ICamera::SphericalTargetState& sphericalState,
    std::vector<CVirtualGimbalEvent>& out) const
{
    if (!camera)
        return false;

    const auto effectiveTarget = target.hasTargetPosition ? target.targetPosition : sphericalState.target;
    ICamera::PathState currentState = {};
    const ICamera::PathState* currentStateOverride = camera->tryGetPathState(currentState) ? &currentState : nullptr;
    ICamera::PathStateLimits pathLimits = CCameraPathUtilities::makeDefaultPathLimits();
    camera->tryGetPathStateLimits(pathLimits);
    SCameraPathStateTransition transition = {};
    if (!CCameraPathUtilities::tryBuildPathStateTransition(
            effectiveTarget,
            camera->getGimbal().getPosition(),
            target.position,
            pathLimits,
            currentStateOverride,
            target.hasPathState ? &target.pathState : nullptr,
            transition))
    {
        return false;
    }

    const auto moveDenom = getMoveMagnitudeDenominator(camera);
    const auto rotationDenom = getRotationMagnitudeDenominator(camera);
    appendPathDeltaEvents(out, transition.delta, moveDenom, rotationDenom);
    return !out.empty();
}

bool CCameraGoalSolver::buildSphericalEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
{
    ICamera::SphericalTargetState sphericalState;
    if (!camera || !camera->tryGetSphericalTargetState(sphericalState))
        return false;

    if (camera->getKind() == ICamera::CameraKind::Path)
        return buildPathEvents(camera, target, sphericalState, out);

    SCameraTargetRelativeState goal;
    if (!CCameraGoalUtilities::tryResolveCanonicalTargetRelativeState(target, sphericalState, goal))
        return false;

    switch (camera->getKind())
    {
        case ICamera::CameraKind::Orbit:
        case ICamera::CameraKind::DollyZoom:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::OrbitTranslatePolicy);

        case ICamera::CameraKind::Turntable:
        case ICamera::CameraKind::Arcball:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::RotateDistancePolicy);

        case ICamera::CameraKind::TopDown:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::TopDownPolicy);

        case ICamera::CameraKind::Isometric:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::IsometricPolicy);

        case ICamera::CameraKind::Dolly:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::DollyPolicy);

        case ICamera::CameraKind::Chase:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::ChasePolicy);

        default:
            return buildTargetRelativeEvents(camera, sphericalState, goal, out, SCameraTargetRelativeRigDefaults::OrbitTranslatePolicy);
    }
}

bool CCameraGoalSolver::buildFreeEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const
{
    const ICamera::CGimbal& gimbal = camera->getGimbal();
    const hlsl::float64_t3 currentPos = gimbal.getPosition();
    const hlsl::float64_t3 deltaWorld = target.position - currentPos;
    CCameraVirtualEventUtilities::appendWorldTranslationAsLocalEvents(
        out,
        gimbal.getOrientation(),
        deltaWorld,
        SGoalSolverDefaults::UnitAxisDenominator,
        SGoalSolverDefaults::ScalarToleranceVec);

    switch (camera->getKind())
    {
        case ICamera::CameraKind::FPS:
        {
            const hlsl::float64_t2 currentPitchYaw = hlsl::CCameraMathUtilities::getPitchYawFromOrientation(gimbal.getOrientation());
            const hlsl::float64_t2 targetPitchYaw = hlsl::CCameraMathUtilities::getPitchYawFromOrientation(target.orientation);

            const double rotScale = camera->getRotationSpeedScale();
            const double invScale = rotScale == 0.0 ? SGoalSolverDefaults::UnitScale : (SGoalSolverDefaults::UnitScale / rotScale);

            appendYawPitchRollEvents(
                out,
                hlsl::float64_t3(
                    hlsl::CCameraMathUtilities::wrapAngleRad<hlsl::float64_t>(targetPitchYaw.x - currentPitchYaw.x) * invScale,
                    hlsl::CCameraMathUtilities::wrapAngleRad<hlsl::float64_t>(targetPitchYaw.y - currentPitchYaw.y) * invScale,
                    0.0),
                SGoalSolverDefaults::UnitScale,
                false);
        } break;

        case ICamera::CameraKind::Free:
        {
            appendYawPitchRollEvents(
                out,
                hlsl::CCameraMathUtilities::getOrientationDeltaEulerRadiansYXZ<hlsl::float64_t>(gimbal.getOrientation(), target.orientation),
                SGoalSolverDefaults::UnitScale);
        } break;

        default:
            break;
    }

    return !out.empty();
}

} // namespace nbl::core
