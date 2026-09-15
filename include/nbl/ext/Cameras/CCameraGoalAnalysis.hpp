// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_GOAL_ANALYSIS_HPP_
#define _C_CAMERA_GOAL_ANALYSIS_HPP_

#include "CCameraPreset.hpp"
#include "CCameraGoalSolver.hpp"

namespace nbl::core
{

/// @brief Reusable typed answer for `goal/preset -> camera` compatibility checks.
struct SCameraGoalApplyAnalysis
{
    CCameraGoal goal = {};
    CCameraGoalSolver::SCompatibilityResult compatibility = {};
    bool hasCamera = false;
    bool finiteGoal = false;
    bool canApply = false;

    inline bool exact() const
    {
        return compatibility.exact;
    }

    inline bool dropsGoalState() const
    {
        return compatibility.missingGoalStateMask != ICamera::GoalStateNone;
    }

    inline bool usesSharedStateOnly() const
    {
        return !compatibility.sameKind && goal.sourceKind != ICamera::CameraKind::Unknown && !dropsGoalState();
    }

    inline bool isMeaningfulApply() const
    {
        return canApply;
    }
};

/// @brief Reusable typed answer for `camera -> goal` capture viability.
struct SCameraCaptureAnalysis
{
    CCameraGoal goal = {};
    bool hasCamera = false;
    bool capturedGoal = false;
    bool finiteGoal = false;
    bool canCapture = false;
};

struct CCameraGoalAnalysisUtilities final
{
public:
    static inline SCameraGoalApplyAnalysis analyzeGoalApply(const CCameraGoalSolver& solver, const ICamera* camera, const CCameraGoal& goal)
    {
        SCameraGoalApplyAnalysis analysis;
        analysis.goal = CCameraGoalUtilities::canonicalizeGoal(goal);
        analysis.hasCamera = camera != nullptr;
        analysis.finiteGoal = CCameraGoalUtilities::isGoalFinite(analysis.goal);
        analysis.canApply = analysis.hasCamera && analysis.finiteGoal;
        if (analysis.hasCamera)
            analysis.compatibility = solver.analyzeCompatibility(camera, analysis.goal);
        return analysis;
    }

    static inline SCameraGoalApplyAnalysis analyzePresetApply(const CCameraGoalSolver& solver, const ICamera* camera, const CCameraPreset& preset)
    {
        return analyzeGoalApply(solver, camera, CCameraPresetUtilities::makeGoalFromPreset(preset));
    }

    static inline SCameraCaptureAnalysis analyzeCameraCapture(const CCameraGoalSolver& solver, ICamera* camera)
    {
        SCameraCaptureAnalysis analysis;
        const auto capture = solver.captureDetailed(camera);
        analysis.goal = capture.goal;
        analysis.hasCamera = capture.hasCamera;
        analysis.capturedGoal = capture.captured;
        analysis.finiteGoal = capture.finiteGoal;
        analysis.canCapture = capture.canUseGoal();
        return analysis;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_GOAL_ANALYSIS_HPP_
