// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_FLOW_HPP_
#define _C_CAMERA_PRESET_FLOW_HPP_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "CCameraGoalAnalysis.hpp"

namespace nbl::core
{

/// @brief Reusable aggregate summary for applying one preset to multiple cameras.
struct SCameraPresetApplySummary
{
    uint32_t targetCount = 0u;
    uint32_t successCount = 0u;
    uint32_t approximateCount = 0u;
    uint32_t failureCount = 0u;

    inline bool hasTargets() const
    {
        return targetCount > 0u;
    }

    inline bool succeeded() const
    {
        return hasTargets() && failureCount == 0u;
    }

    inline bool approximate() const
    {
        return approximateCount > 0u;
    }
};

struct CCameraPresetFlowUtilities final
{
    /// @brief Compare the current camera state against a preset using the shared goal representation.
    static inline bool comparePresetToCameraState(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset,
        const double posEps, const double rotEpsDeg, const double scalarEps)
    {
        const auto capture = solver.captureDetailed(camera);
        if (!capture.canUseGoal())
            return false;

        return CCameraGoalUtilities::compareGoals(
            capture.goal,
            CCameraPresetUtilities::makeGoalFromPreset(preset),
            posEps,
            rotEpsDeg,
            scalarEps);
    }

    /// @brief Explain the first visible mismatch between a camera state and a preset.
    static inline std::string describePresetCameraMismatch(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset)
    {
        const auto capture = solver.captureDetailed(camera);
        if (!capture.hasCamera)
            return "camera=null";
        if (!capture.captured)
            return "goal_state=unavailable";
        if (!capture.finiteGoal)
            return "goal_state=invalid";

        return CCameraGoalUtilities::describeGoalMismatch(capture.goal, CCameraPresetUtilities::makeGoalFromPreset(preset));
    }

    /// @brief Build a preset from an already analyzed capture result.
    static inline bool tryCapturePreset(const SCameraCaptureAnalysis& captureAnalysis, ICamera* camera, std::string_view name, CCameraPreset& preset)
    {
        preset = {};
        preset.name = std::string(name);
        if (!captureAnalysis.canCapture || !camera)
            return false;

        preset.identifier = std::string(camera->getIdentifier());
        CCameraPresetUtilities::assignGoalToPreset(preset, captureAnalysis.goal);
        return true;
    }

    /// @brief Capture a preset directly from a camera through the shared goal solver.
    static inline bool tryCapturePreset(const CCameraGoalSolver& solver, ICamera* camera, std::string_view name, CCameraPreset& preset)
    {
        return tryCapturePreset(CCameraGoalAnalysisUtilities::analyzeCameraCapture(solver, camera), camera, name, preset);
    }

    /// @brief Value-returning convenience wrapper around `tryCapturePreset`.
    static inline CCameraPreset capturePreset(const CCameraGoalSolver& solver, ICamera* camera, std::string_view name)
    {
        CCameraPreset preset;
        tryCapturePreset(solver, camera, name, preset);
        return preset;
    }

    /// @brief Apply a preset through the shared goal solver and preserve detailed apply diagnostics.
    static inline CCameraGoalSolver::SApplyResult applyPresetDetailed(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset)
    {
        if (!camera)
            return {};

        return solver.applyDetailed(camera, CCameraPresetUtilities::makeGoalFromPreset(preset));
    }

    /// @brief Bool-returning convenience wrapper around `applyPresetDetailed`.
    static inline bool applyPreset(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset)
    {
        return applyPresetDetailed(solver, camera, preset).succeeded();
    }

    /// @brief Fold one detailed apply result into an aggregate preset-apply summary.
    static inline void accumulatePresetApplySummary(SCameraPresetApplySummary& summary, const CCameraGoalSolver::SApplyResult& result)
    {
        ++summary.targetCount;
        if (result.succeeded())
        {
            ++summary.successCount;
            if (result.approximate())
                ++summary.approximateCount;
        }
        else
        {
            ++summary.failureCount;
        }
    }

    /// @brief Apply one preset to a camera range and collect a typed aggregate summary.
    static inline SCameraPresetApplySummary applyPresetToCameraRange(const CCameraGoalSolver& solver, std::span<ICamera* const> cameras, const CCameraPreset& preset)
    {
        SCameraPresetApplySummary summary;
        for (auto* camera : cameras)
        {
            if (!camera)
                continue;

            accumulatePresetApplySummary(summary, applyPresetDetailed(solver, camera, preset));
        }

        return summary;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_PRESET_FLOW_HPP_
