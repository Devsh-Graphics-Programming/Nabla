// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_
#define _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "CCameraFollowRegressionUtilities.hpp"
#include "CCameraScriptedRuntime.hpp"
#include "SCameraTypes.hpp"

namespace nbl::ext::cameras
{

/// @brief Runtime state for authored scripted checks.
///
/// This state stores:
/// - the index of the next authored check to evaluate
/// - one baseline pose reference
/// - one step pose reference
struct CCameraScriptedCheckRuntimeState
{
    struct SPoseReference final : SCameraRigPose
    {
        bool valid = false;
    };

    size_t nextCheckIndex = 0u;
    SPoseReference baseline = {};
    SPoseReference step = {};
};

/// @brief Shared per-frame evaluation context for authored scripted checks.
struct CCameraScriptedCheckContext
{
    uint64_t frame = 0ull;
    ICamera* camera = nullptr;
    const CVirtualGimbalEvent* imguizmoVirtual = nullptr;
    uint32_t imguizmoVirtualCount = 0u;
    const CTrackedTarget* trackedTarget = nullptr;
    const SCameraFollowConfig* followConfig = nullptr;
    const SCameraProjectionContext* followProjectionContext = nullptr;
    const CCameraGoalSolver* goalSolver = nullptr;
};

/// @brief Reusable log entry produced by scripted check evaluation.
struct CCameraScriptedCheckLogEntry
{
    bool failure = false;
    std::string text;
};

/// @brief Result for one frame worth of scripted checks.
struct CCameraScriptedCheckFrameResult
{
    std::vector<CCameraScriptedCheckLogEntry> logs;
    bool hadFailures = false;
};

struct CCameraScriptedCheckRunnerUtilities final
{
    static void scriptedCheckSetStepReference(
        CCameraScriptedCheckRuntimeState& state,
        const hlsl::float64_t3& position,
        const hlsl::math::quaternion<hlsl::float64_t>& orientation);
    static void scriptedCheckSetBaselineReference(
        CCameraScriptedCheckRuntimeState& state,
        const hlsl::float64_t3& position,
        const hlsl::math::quaternion<hlsl::float64_t>& orientation);
    static bool scriptedCheckComputePoseDelta(
        const hlsl::float64_t3& currentPosition,
        const hlsl::math::quaternion<hlsl::float64_t>& currentOrientation,
        const hlsl::float64_t3& referencePosition,
        const hlsl::math::quaternion<hlsl::float64_t>& referenceOrientation,
        SCameraPoseDelta<hlsl::float64_t>& outDelta);

    template<typename Fn>
    static inline std::string buildScriptedCheckMessage(Fn&& formatter)
    {
        std::ostringstream oss;
        formatter(oss);
        return oss.str();
    }

    static void appendScriptedCheckLog(
        CCameraScriptedCheckFrameResult& result,
        bool failure,
        std::string&& text);

    /// @brief Evaluate all authored scripted checks scheduled for the current frame.
    static CCameraScriptedCheckFrameResult evaluateScriptedChecksForFrame(
        const std::vector<CCameraScriptedInputCheck>& checks,
        CCameraScriptedCheckRuntimeState& state,
        const CCameraScriptedCheckContext& context);
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_SCRIPTED_CHECK_RUNNER_HPP_
