// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

#include <string>

#include "CCameraFollowUtilities.hpp"

namespace nbl::system
{

struct SCameraProjectedTargetMetrics final
{
    hlsl::float32_t2 ndc = hlsl::float32_t2(0.0f);
    float radius = 0.0f;
};

/// @brief Reusable follow validation helpers.
///
/// The checks stay camera-domain:
///
/// - camera-to-target direction must match the camera forward axis for locking modes
/// - target distance must be finite and internally consistent
/// - spherical cameras must write the tracked target back into spherical target state
/// - spherical distance must match the goal-derived distance when present
struct SCameraFollowRegressionResult
{
    bool passed = false;
    bool hasLockMetrics = false;
    float lockAngleDeg = 0.0f;
    double targetDistance = 0.0;
    bool hasProjectedMetrics = false;
    SCameraProjectedTargetMetrics projectedTarget = {};
    bool hasSphericalState = false;
    hlsl::float64_t3 sphericalTarget = hlsl::float64_t3(0.0);
    float sphericalDistance = 0.0f;
};

/// @brief Reusable visual/debug metrics for one active follow configuration.
struct SCameraFollowVisualMetrics
{
    bool active = false;
    core::ECameraFollowMode mode = core::ECameraFollowMode::Disabled;
    bool lockValid = false;
    float lockAngleDeg = 0.0f;
    float targetDistance = 0.0f;
    bool projectedValid = false;
    SCameraProjectedTargetMetrics projectedTarget = {};
};

/// @brief Shared view/projection bundle for CPU-side projected target metrics.
struct SCameraProjectionContext
{
    hlsl::float32_t4x4 viewMatrix = hlsl::float32_t4x4(1.0f);
    hlsl::float32_t4x4 projectionMatrix = hlsl::float32_t4x4(1.0f);
};

/// @brief Shared tolerances for follow target lock, writeback, and projected-center checks.
struct SCameraFollowRegressionThresholds
{
    static inline constexpr float DefaultClipWEpsilon = 1e-5f;
    static inline constexpr float DefaultProjectedNdcTolerance = 0.03f;
    static inline constexpr float DefaultLockAngleToleranceDeg = static_cast<float>(core::SCameraToolingThresholds::DefaultAngularToleranceDeg);
    static inline constexpr double DefaultDistanceTolerance = core::SCameraToolingThresholds::ScalarTolerance;
    static inline constexpr double DefaultTargetTolerance = core::SCameraToolingThresholds::TinyScalarEpsilon;
    static inline constexpr double DefaultPositionTolerance = core::SCameraToolingThresholds::DefaultPositionTolerance;
    static inline constexpr double DefaultRotationToleranceDeg = core::SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static inline constexpr double DefaultScalarTolerance = core::SCameraToolingThresholds::ScalarTolerance;

    float clipWEpsilon = DefaultClipWEpsilon;
    float projectedNdcTolerance = DefaultProjectedNdcTolerance;
    float lockAngleToleranceDeg = DefaultLockAngleToleranceDeg;
    double distanceTolerance = DefaultDistanceTolerance;
    double targetTolerance = DefaultTargetTolerance;
    double positionTolerance = DefaultPositionTolerance;
    double rotationToleranceDeg = DefaultRotationToleranceDeg;
    double scalarTolerance = DefaultScalarTolerance;
};

/// @brief Bundled reusable follow regression flow.
/// The helper builds a follow goal, applies it, verifies the resulting camera state,
/// and then checks lock/writeback follow consistency.
struct SCameraFollowApplyValidationResult
{
    bool hasGoal = false;
    core::CCameraGoal goal = {};
    core::CCameraGoalSolver::SApplyResult applyResult = {};
    bool hasCapturedGoal = false;
    core::CCameraGoal capturedGoal = {};
    SCameraFollowRegressionResult regression = {};
};

struct CCameraFollowRegressionUtilities final
{
public:
    static SCameraFollowRegressionThresholds makeFollowRegressionThresholds(
        float projectedNdcTolerance = SCameraFollowRegressionThresholds::DefaultProjectedNdcTolerance,
        float lockAngleToleranceDeg = SCameraFollowRegressionThresholds::DefaultLockAngleToleranceDeg);

    static bool tryComputeProjectedFollowTargetMetrics(
        const SCameraProjectionContext& projectionContext,
        const core::CTrackedTarget& trackedTarget,
        SCameraProjectedTargetMetrics& outMetrics,
        float clipWEpsilon = SCameraFollowRegressionThresholds::DefaultClipWEpsilon);

    static bool validateProjectedFollowTargetContract(
        const SCameraProjectionContext& projectionContext,
        const core::CTrackedTarget& trackedTarget,
        SCameraProjectedTargetMetrics& outMetrics,
        std::string* error = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});

    static SCameraFollowVisualMetrics buildFollowVisualMetrics(
        core::ICamera* camera,
        const core::CTrackedTarget& trackedTarget,
        const core::SCameraFollowConfig* followConfig,
        const SCameraProjectionContext* projectionContext = nullptr);

    static bool validateFollowTargetContract(
        core::ICamera* camera,
        const core::CTrackedTarget& trackedTarget,
        const core::SCameraFollowConfig& followConfig,
        const core::CCameraGoal& followGoal,
        SCameraFollowRegressionResult& out,
        std::string* error = nullptr,
        const SCameraProjectionContext* projectionContext = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});

    static bool buildApplyAndValidateFollowTargetContract(
        const core::CCameraGoalSolver& solver,
        core::ICamera* camera,
        const core::CTrackedTarget& trackedTarget,
        const core::SCameraFollowConfig& followConfig,
        SCameraFollowApplyValidationResult& out,
        std::string* error = nullptr,
        const SCameraProjectionContext* projectionContext = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});
};

} // namespace nbl::system

#endif // _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

