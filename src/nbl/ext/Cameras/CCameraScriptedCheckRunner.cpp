// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraScriptedCheckRunner.hpp"

namespace nbl::system
{

void CCameraScriptedCheckRunnerUtilities::scriptedCheckSetStepReference(
    CCameraScriptedCheckRuntimeState& state,
    const hlsl::float64_t3& position,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
{
    state.step.valid = true;
    state.step.position = position;
    state.step.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(orientation);
}

void CCameraScriptedCheckRunnerUtilities::scriptedCheckSetBaselineReference(
    CCameraScriptedCheckRuntimeState& state,
    const hlsl::float64_t3& position,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
{
    state.baseline.valid = true;
    state.baseline.position = position;
    state.baseline.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(orientation);
    scriptedCheckSetStepReference(state, position, orientation);
}

bool CCameraScriptedCheckRunnerUtilities::scriptedCheckComputePoseDelta(
    const hlsl::float64_t3& currentPosition,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& currentOrientation,
    const hlsl::float64_t3& referencePosition,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& referenceOrientation,
    hlsl::SCameraPoseDelta<hlsl::float64_t>& outDelta)
{
    return hlsl::CCameraMathUtilities::tryComputePoseDelta(
        currentPosition,
        currentOrientation,
        referencePosition,
        referenceOrientation,
        outDelta);
}

void CCameraScriptedCheckRunnerUtilities::appendScriptedCheckLog(
    CCameraScriptedCheckFrameResult& result,
    const bool failure,
    std::string&& text)
{
    result.logs.push_back({
        .failure = failure,
        .text = std::move(text)
    });
    result.hadFailures = result.hadFailures || failure;
}

CCameraScriptedCheckFrameResult CCameraScriptedCheckRunnerUtilities::evaluateScriptedChecksForFrame(
    const std::vector<CCameraScriptedInputCheck>& checks,
    CCameraScriptedCheckRuntimeState& state,
    const CCameraScriptedCheckContext& context)
{
    CCameraScriptedCheckFrameResult result = {};

    while (state.nextCheckIndex < checks.size() && checks[state.nextCheckIndex].frame == context.frame)
    {
        const auto& check = checks[state.nextCheckIndex];

        if (!context.camera)
        {
            appendScriptedCheckLog(
                result,
                true,
                buildScriptedCheckMessage([&](std::ostringstream& oss)
                {
                    oss << "[script][fail] check frame=" << context.frame << " no active camera";
                }));
            ++state.nextCheckIndex;
            continue;
        }

        const auto& gimbal = context.camera->getGimbal();
        const auto pos = gimbal.getPosition();
        const auto orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(gimbal.getOrientation());
        const auto eulerDeg = hlsl::CCameraMathUtilities::castVector<hlsl::float32_t>(hlsl::CCameraMathUtilities::getCameraOrientationEulerDegrees(orientation));

        if (!hlsl::CCameraMathUtilities::isFiniteVec3(pos) || !hlsl::CCameraMathUtilities::isFiniteQuaternion(orientation) || !hlsl::CCameraMathUtilities::isFiniteVec3(eulerDeg))
        {
            appendScriptedCheckLog(
                result,
                true,
                buildScriptedCheckMessage([&](std::ostringstream& oss)
                {
                    oss << "[script][fail] check frame=" << context.frame << " non-finite gimbal state";
                }));
            ++state.nextCheckIndex;
            continue;
        }

        switch (check.kind)
        {
            case CCameraScriptedInputCheck::Kind::Baseline:
            {
                scriptedCheckSetBaselineReference(state, pos, orientation);
                appendScriptedCheckLog(
                    result,
                    false,
                    buildScriptedCheckMessage([&](std::ostringstream& oss)
                    {
                        oss << std::fixed << std::setprecision(3);
                        oss << "[script][pass] baseline frame=" << context.frame
                            << " pos=(" << pos.x << ", " << pos.y << ", " << pos.z << ")"
                            << " euler_deg=(" << eulerDeg.x << ", " << eulerDeg.y << ", " << eulerDeg.z << ")";
                    }));
                break;
            }
            case CCameraScriptedInputCheck::Kind::ImguizmoVirtual:
            {
                bool ok = true;
                if (!context.imguizmoVirtual || context.imguizmoVirtualCount == 0u)
                {
                    ok = false;
                }
                else
                {
                    for (const auto& expected : check.expectedVirtualEvents)
                    {
                        bool found = false;
                        double actual = 0.0;
                        for (uint32_t i = 0u; i < context.imguizmoVirtualCount; ++i)
                        {
                            if (context.imguizmoVirtual[i].type == expected.type)
                            {
                                found = true;
                                actual = context.imguizmoVirtual[i].magnitude;
                                break;
                            }
                        }

                        if (!found || hlsl::abs(actual - expected.magnitude) > check.tolerance)
                        {
                            ok = false;
                            appendScriptedCheckLog(
                                result,
                                true,
                                buildScriptedCheckMessage([&](std::ostringstream& oss)
                                {
                                    oss << std::fixed << std::setprecision(6);
                                    oss << "[script][fail] imguizmo_virtual frame=" << context.frame
                                        << " type=" << core::CVirtualGimbalEvent::virtualEventToString(expected.type).data()
                                        << " expected=" << expected.magnitude
                                        << " actual=" << actual
                                        << " tol=" << check.tolerance;
                                }));
                        }
                    }
                }

                if (ok)
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][pass] imguizmo_virtual frame=" << context.frame
                                << " events=" << check.expectedVirtualEvents.size();
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalNear:
            {
                bool ok = true;
                if (check.hasExpectedPos)
                {
                    const double distance = hlsl::length(pos - hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(check.expectedPos));
                    if (distance > check.posTolerance)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_near frame=" << context.frame
                                    << " pos_diff=" << distance
                                    << " tol=" << check.posTolerance;
                            }));
                    }
                }
                if (check.hasExpectedEuler)
                {
                    const auto expectedOrientation = hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(
                        hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(check.expectedEulerDeg));
                    hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
                    if (!scriptedCheckComputePoseDelta(pos, orientation, pos, expectedOrientation, poseDelta))
                        poseDelta.rotationDeg = std::numeric_limits<hlsl::float64_t>::infinity();
                    const auto rotationDeltaDeg = poseDelta.rotationDeg;
                    if (rotationDeltaDeg > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_near frame=" << context.frame
                                    << " rot_delta_deg=" << rotationDeltaDeg
                                    << " tol=" << check.eulerToleranceDeg;
                            }));
                    }
                }

                if (ok)
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][pass] gimbal_near frame=" << context.frame;
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalDelta:
            {
                if (!state.baseline.valid)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] gimbal_delta frame=" << context.frame << " missing baseline";
                        }));
                    break;
                }

                hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
                if (!scriptedCheckComputePoseDelta(pos, orientation, state.baseline.position, state.baseline.orientation, poseDelta))
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] gimbal_delta frame=" << context.frame << " non-finite pose delta";
                        }));
                    break;
                }

                if (poseDelta.position > check.posTolerance || poseDelta.rotationDeg > check.eulerToleranceDeg)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][fail] gimbal_delta frame=" << context.frame
                                << " pos_diff=" << poseDelta.position
                                << " tol=" << check.posTolerance
                                << " rot_delta_deg=" << poseDelta.rotationDeg
                                << " tol=" << check.eulerToleranceDeg;
                        }));
                }
                else
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][pass] gimbal_delta frame=" << context.frame
                                << " pos_diff=" << poseDelta.position
                                << " rot_delta_deg=" << poseDelta.rotationDeg;
                        }));
                }
                break;
            }
            case CCameraScriptedInputCheck::Kind::GimbalStep:
            {
                if (!state.step.valid)
                {
                    if (state.baseline.valid)
                    {
                        scriptedCheckSetStepReference(state, state.baseline.position, state.baseline.orientation);
                    }
                    else
                    {
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << "[script][fail] gimbal_step frame=" << context.frame << " missing step reference";
                            }));
                        scriptedCheckSetStepReference(state, pos, orientation);
                        ++state.nextCheckIndex;
                        continue;
                    }
                }

                hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
                if (!scriptedCheckComputePoseDelta(pos, orientation, state.step.position, state.step.orientation, poseDelta))
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] gimbal_step frame=" << context.frame << " non-finite pose delta";
                        }));
                    scriptedCheckSetStepReference(state, pos, orientation);
                    break;
                }

                bool ok = true;
                bool requiresProgress = false;
                bool hasProgress = false;
                if (check.hasPosDeltaConstraint)
                {
                    if (poseDelta.position > check.posTolerance)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " pos_delta=" << poseDelta.position
                                    << " max=" << check.posTolerance;
                            }));
                    }
                    if (check.minPosDelta > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || poseDelta.position >= check.minPosDelta;
                    }
                }
                if (check.hasEulerDeltaConstraint)
                {
                    if (poseDelta.rotationDeg > check.eulerToleranceDeg)
                    {
                        ok = false;
                        appendScriptedCheckLog(
                            result,
                            true,
                            buildScriptedCheckMessage([&](std::ostringstream& oss)
                            {
                                oss << std::fixed << std::setprecision(6);
                                oss << "[script][fail] gimbal_step frame=" << context.frame
                                    << " rot_delta_deg=" << poseDelta.rotationDeg
                                    << " max=" << check.eulerToleranceDeg;
                            }));
                    }
                    if (check.minEulerDeltaDeg > 0.0f)
                    {
                        requiresProgress = true;
                        hasProgress = hasProgress || poseDelta.rotationDeg >= check.minEulerDeltaDeg;
                    }
                }
                if (requiresProgress && !hasProgress)
                {
                    ok = false;
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][fail] gimbal_step frame=" << context.frame
                                << " missing progress pos_delta=" << poseDelta.position
                                << " rot_delta_deg=" << poseDelta.rotationDeg;
                        }));
                }

                if (ok)
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][pass] gimbal_step frame=" << context.frame
                                << " pos_delta=" << poseDelta.position
                                << " rot_delta_deg=" << poseDelta.rotationDeg;
                        }));
                }
                scriptedCheckSetStepReference(state, pos, orientation);
                break;
            }
            case CCameraScriptedInputCheck::Kind::FollowTargetLock:
            {
                if (!context.followConfig)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << " missing follow config";
                        }));
                    break;
                }
                if (!context.trackedTarget)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << " missing tracked target";
                        }));
                    break;
                }
                if (!context.goalSolver)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << " missing goal solver";
                        }));
                    break;
                }

                SCameraFollowRegressionResult regression = {};
                std::string regressionError;
                core::CCameraGoal expectedFollowGoal = {};
                const auto thresholds = CCameraFollowRegressionUtilities::makeFollowRegressionThresholds(check.posTolerance, check.eulerToleranceDeg);
                const bool ok = core::CCameraFollowUtilities::tryBuildFollowGoal(
                        *context.goalSolver,
                        context.camera,
                        *context.trackedTarget,
                        *context.followConfig,
                        expectedFollowGoal) &&
                    CCameraFollowRegressionUtilities::validateFollowTargetContract(
                        context.camera,
                        *context.trackedTarget,
                        *context.followConfig,
                        expectedFollowGoal,
                        regression,
                        &regressionError,
                        context.followProjectionContext,
                        thresholds);

                if (!ok)
                {
                    appendScriptedCheckLog(
                        result,
                        true,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << "[script][fail] follow_lock frame=" << context.frame << ' '
                                << (regressionError.empty() ? "follow validation mismatch" : regressionError);
                        }));
                }
                else
                {
                    appendScriptedCheckLog(
                        result,
                        false,
                        buildScriptedCheckMessage([&](std::ostringstream& oss)
                        {
                            oss << std::fixed << std::setprecision(6);
                            oss << "[script][pass] follow_lock frame=" << context.frame
                                << " angle_deg=" << regression.lockAngleDeg
                                << " target_distance=" << regression.targetDistance
                                << " screen_ndc=" << regression.projectedTarget.radius;
                        }));
                }
                break;
            }
        }

        ++state.nextCheckIndex;
    }

    return result;
}

} // namespace nbl::system
