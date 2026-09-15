// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESENTATION_UTILITIES_HPP_
#define _C_CAMERA_PRESENTATION_UTILITIES_HPP_

#include <string>

#include "CCameraTextUtilities.hpp"

namespace nbl::ui
{

/// @brief Shared exactness-oriented filter used by preset presentation surfaces.
enum class EPresetApplyPresentationFilter : uint8_t
{
    All,
    Exact,
    BestEffort
};

/// @brief Shared badge/pill policy derived from one analyzed presentation answer.
struct SCameraGoalApplyPresentationBadges final
{
    bool exact = false;
    bool bestEffort = false;
    bool dropsState = false;
    bool sharedStateOnly = false;
    bool blocked = false;
};

/// @brief Presentation-ready wrapper around analyzed goal apply compatibility.
struct SCameraGoalApplyPresentation final : core::SCameraGoalApplyAnalysis
{
    SCameraGoalApplyPresentationBadges badges;
    std::string sourceKindLabel;
    std::string goalStateLabel;
    std::string compatibilityLabel;
    std::string policyLabel;

    inline bool matchesFilter(const EPresetApplyPresentationFilter mode) const
    {
        switch (mode)
        {
            case EPresetApplyPresentationFilter::All:
                return true;
            case EPresetApplyPresentationFilter::Exact:
                return hasCamera && exact();
            case EPresetApplyPresentationFilter::BestEffort:
                return hasCamera && !exact();
            default:
                return true;
        }
    }
};

/// @brief Presentation-ready wrapper around analyzed camera capture viability.
struct SCameraCapturePresentation final : core::SCameraCaptureAnalysis
{
    std::string policyLabel;
};

struct CCameraPresentationUtilities final
{
    /// @brief Shared user-facing label for the exactness filter selector.
    static inline const char* getPresetApplyPresentationFilterLabel(const EPresetApplyPresentationFilter mode)
    {
        switch (mode)
        {
            case EPresetApplyPresentationFilter::All:
                return "All";
            case EPresetApplyPresentationFilter::Exact:
                return "Exact";
            case EPresetApplyPresentationFilter::BestEffort:
                return "Best-effort";
            default:
                return "All";
        }
    }

    /// @brief Build reusable badge flags for one preset/keyframe compatibility answer.
    static inline SCameraGoalApplyPresentationBadges collectGoalApplyPresentationBadges(const SCameraGoalApplyPresentation& presentation)
    {
        SCameraGoalApplyPresentationBadges badges;
        badges.exact = presentation.exact();
        badges.bestEffort = presentation.hasCamera && !presentation.exact();
        badges.dropsState = presentation.dropsGoalState();
        badges.sharedStateOnly = presentation.usesSharedStateOnly();
        badges.blocked = !presentation.canApply;
        return badges;
    }

    /// @brief Build presentation text for one analyzed goal-apply result.
    static inline SCameraGoalApplyPresentation makeGoalApplyPresentation(const core::SCameraGoalApplyAnalysis& analysis, const core::ICamera* targetCamera)
    {
        SCameraGoalApplyPresentation presentation;
        static_cast<core::SCameraGoalApplyAnalysis&>(presentation) = analysis;
        presentation.badges = collectGoalApplyPresentationBadges(presentation);
        presentation.sourceKindLabel = std::string(CCameraTextUtilities::getCameraTypeLabel(presentation.goal.sourceKind));
        presentation.goalStateLabel = CCameraTextUtilities::describeGoalStateMask(presentation.goal.sourceGoalStateMask);
        presentation.compatibilityLabel = CCameraTextUtilities::describeGoalApplyCompatibility(analysis, targetCamera);
        presentation.policyLabel = CCameraTextUtilities::describeGoalApplyPolicy(analysis);
        return presentation;
    }

    /// @brief Analyze one preset against one camera and return reusable presentation data.
    static inline SCameraGoalApplyPresentation analyzePresetPresentation(const core::CCameraGoalSolver& solver, const core::ICamera* camera, const core::CCameraPreset& preset)
    {
        return makeGoalApplyPresentation(core::CCameraGoalAnalysisUtilities::analyzePresetApply(solver, camera, preset), camera);
    }

    /// @brief Analyze one camera capture path and return reusable presentation data.
    static inline SCameraCapturePresentation analyzeCapturePresentation(const core::CCameraGoalSolver& solver, core::ICamera* camera)
    {
        SCameraCapturePresentation presentation;
        static_cast<core::SCameraCaptureAnalysis&>(presentation) = core::CCameraGoalAnalysisUtilities::analyzeCameraCapture(solver, camera);
        presentation.policyLabel = CCameraTextUtilities::describeCameraCapturePolicy(presentation, camera);
        return presentation;
    }
};

} // namespace nbl::ui

#endif // _C_CAMERA_PRESENTATION_UTILITIES_HPP_
