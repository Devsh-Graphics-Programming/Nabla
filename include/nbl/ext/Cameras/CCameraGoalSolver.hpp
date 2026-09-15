#ifndef _C_CAMERA_GOAL_SOLVER_HPP_
#define _C_CAMERA_GOAL_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "CCameraGoal.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraVirtualEventUtilities.hpp"
#include "nbl/core/util/bitflag.h"

namespace nbl::core
{

/// @brief Goal capture, compatibility analysis, and goal application helper.
///
/// The solver captures canonical state into `CCameraGoal`, compares a goal
/// against one target camera, applies typed fragments directly when the camera
/// exposes them, and builds virtual-event replay when a typed fragment must be
/// approximated through `manipulate(...)`.
class CCameraGoalSolver
{
public:
    /// @brief Detailed result returned by one goal-capture attempt.
    struct SCaptureResult
    {
        bool hasCamera = false;
        bool captured = false;
        bool finiteGoal = false;
        CCameraGoal goal = {};

        inline bool canUseGoal() const
        {
            return hasCamera && captured && finiteGoal;
        }
    };

    /// @brief Compatibility of a goal with a target camera kind and state mask.
    struct SCompatibilityResult
    {
        bool sameKind = false;
        bool exact = false;
        ICamera::goal_state_flags_t requiredGoalStateMask = ICamera::GoalStateNone;
        ICamera::goal_state_flags_t supportedGoalStateMask = ICamera::GoalStateNone;
        ICamera::goal_state_flags_t missingGoalStateMask = ICamera::GoalStateNone;
    };

    /// @brief Outcome of one goal-application attempt.
    struct SApplyResult
    {
        enum class EStatus : uint8_t
        {
            Unsupported,
            Failed,
            AlreadySatisfied,
            AppliedAbsoluteOnly,
            AppliedVirtualEvents,
            AppliedAbsoluteAndVirtualEvents
        };

        enum class EIssue : uint32_t
        {
            NoIssue = 0u,
            UsedAbsolutePoseFallback = core::createBitmask({ 0 }),
            MissingSphericalTargetState = core::createBitmask({ 1 }),
            MissingPathState = core::createBitmask({ 2 }),
            MissingDynamicPerspectiveState = core::createBitmask({ 3 }),
            VirtualEventReplayFailed = core::createBitmask({ 4 })
        };

        EStatus status = EStatus::Unsupported;
        bool exact = false;
        uint32_t eventCount = 0u;
        core::bitflag<EIssue> issues = EIssue::NoIssue;

        inline bool succeeded() const
        {
            return status != EStatus::Unsupported && status != EStatus::Failed;
        }

        inline bool changed() const
        {
            return status == EStatus::AppliedAbsoluteOnly ||
                status == EStatus::AppliedVirtualEvents ||
                status == EStatus::AppliedAbsoluteAndVirtualEvents;
        }

        inline bool approximate() const
        {
            return succeeded() && !exact;
        }

        inline bool hasIssue(EIssue issue) const
        {
            return issues.hasFlags(issue);
        }
    };

    bool buildEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const;
    bool capture(ICamera* camera, CCameraGoal& out) const;
    SCaptureResult captureDetailed(ICamera* camera) const;
    SCompatibilityResult analyzeCompatibility(const ICamera* camera, const CCameraGoal& target) const;
    SApplyResult applyDetailed(ICamera* camera, const CCameraGoal& target) const;
    bool apply(ICamera* camera, const CCameraGoal& target) const;

private:
    struct SGoalSolverDefaults final
    {
        static constexpr double UnitScale = 1.0;
        static inline const hlsl::float64_t3 UnitAxisDenominator = hlsl::float64_t3(UnitScale);
        static inline const hlsl::float64_t3 ScalarToleranceVec = hlsl::float64_t3(SCameraToolingThresholds::ScalarTolerance);
        static inline const hlsl::float64_t3 AngularToleranceDegVec = hlsl::float64_t3(SCameraToolingThresholds::DefaultAngularToleranceDeg);
    };

    void appendYawPitchRollEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const hlsl::float64_t3& eulerRadians,
        double denominator,
        bool includeRoll = true) const;
    void appendPathDeltaEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraPathDelta& delta,
        double moveDenominator,
        double rotationDenominator) const;
    double getMoveMagnitudeDenominator(const ICamera* camera) const;
    double getRotationMagnitudeDenominator(const ICamera* camera) const;
    bool computePoseMismatch(ICamera* camera, const CCameraGoal& target, double& outPositionDelta, double& outRotationDeltaDeg) const;
    bool tryApplyAbsoluteReferencePose(ICamera* camera, const CCameraGoal& target, bool& outChanged, bool& outExact) const;
    bool buildTargetRelativeEvents(
        ICamera* camera,
        const ICamera::SphericalTargetState& sphericalState,
        const SCameraTargetRelativeState& goal,
        std::vector<CVirtualGimbalEvent>& out,
        const SCameraTargetRelativeEventPolicy& policy) const;
    bool buildPathEvents(
        ICamera* camera,
        const CCameraGoal& target,
        const ICamera::SphericalTargetState& sphericalState,
        std::vector<CVirtualGimbalEvent>& out) const;
    bool buildSphericalEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const;
    bool buildFreeEvents(ICamera* camera, const CCameraGoal& target, std::vector<CVirtualGimbalEvent>& out) const;
};

} // namespace nbl::core

#endif // _C_CAMERA_GOAL_SOLVER_HPP_

