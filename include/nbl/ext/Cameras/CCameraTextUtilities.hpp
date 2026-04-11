// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_TEXT_UTILITIES_HPP_
#define _C_CAMERA_TEXT_UTILITIES_HPP_

#include <sstream>
#include <string>
#include <string_view>

#include "CCameraFollowUtilities.hpp"
#include "CCameraGoalAnalysis.hpp"
#include "CCameraPresetFlow.hpp"

namespace nbl::ui
{

struct CCameraTextUtilities final
{
public:
	/// @brief Return a short human-readable label for a camera kind.
	static inline std::string_view getCameraTypeLabel(const core::ICamera::CameraKind kind)
	{
		return core::CCameraKindUtilities::getCameraKindLabel(kind);
	}

	/// @brief Return a short human-readable label for a concrete camera instance.
	static inline std::string_view getCameraTypeLabel(const core::ICamera* camera)
	{
		return camera ? getCameraTypeLabel(camera->getKind()) : "Unknown";
	}

	/// @brief Return a short human-readable description for a camera kind.
	static inline std::string_view getCameraTypeDescription(const core::ICamera::CameraKind kind)
	{
		return core::CCameraKindUtilities::getCameraKindDescription(kind);
	}

	/// @brief Return a short human-readable description for a concrete camera instance.
	static inline std::string_view getCameraTypeDescription(const core::ICamera* camera)
	{
		return camera ? getCameraTypeDescription(camera->getKind()) : "Unspecified camera behavior";
	}

	/// @brief Return a short human-readable label for a follow mode.
	static inline constexpr const char* getCameraFollowModeLabel(const core::ECameraFollowMode mode)
	{
		switch (mode)
		{
			case core::ECameraFollowMode::Disabled: return "Disabled";
			case core::ECameraFollowMode::OrbitTarget: return "Orbit target";
			case core::ECameraFollowMode::LookAtTarget: return "Look at target";
			case core::ECameraFollowMode::KeepWorldOffset: return "Keep world offset";
			case core::ECameraFollowMode::KeepLocalOffset: return "Keep local offset";
			default: return "Unknown";
		}
	}

	/// @brief Return a short human-readable description for a follow mode.
	static inline constexpr const char* getCameraFollowModeDescription(const core::ECameraFollowMode mode)
	{
		switch (mode)
		{
			case core::ECameraFollowMode::Disabled: return "Follow disabled";
			case core::ECameraFollowMode::OrbitTarget: return "Keep orbit around moving target and keep it centered";
			case core::ECameraFollowMode::LookAtTarget: return "Keep camera position and lock the view onto the target";
			case core::ECameraFollowMode::KeepWorldOffset: return "Move with the target in world offset and keep it centered";
			case core::ECameraFollowMode::KeepLocalOffset: return "Move with the target in target-local offset and keep it centered";
			default: return "Unknown follow mode";
		}
	}

	/// @brief Describe the typed goal-state mask in a stable human-readable format.
	static inline std::string describeGoalStateMask(const core::ICamera::goal_state_flags_t mask)
	{
		if (mask == core::ICamera::GoalStateNone)
			return "Pose only";

		std::string out;
		auto append = [&](const char* label, const core::ICamera::GoalStateMask bit) -> void
		{
			if (!mask.hasFlags(bit))
				return;
			if (!out.empty())
				out += ", ";
			out += label;
		};

		append("Spherical target", core::ICamera::GoalStateSphericalTarget);
		append("Dynamic perspective", core::ICamera::GoalStateDynamicPerspective);
		append("Path rig state", core::ICamera::GoalStatePath);
		return out;
	}

	/// @brief Describe a detailed goal-apply result for logs, smoke tests, and UI summaries.
	static inline std::string describeApplyResult(const core::CCameraGoalSolver::SApplyResult& result)
	{
		std::ostringstream oss;
		oss << "status=";
		switch (result.status)
		{
			case core::CCameraGoalSolver::SApplyResult::EStatus::Unsupported: oss << "Unsupported"; break;
			case core::CCameraGoalSolver::SApplyResult::EStatus::Failed: oss << "Failed"; break;
			case core::CCameraGoalSolver::SApplyResult::EStatus::AlreadySatisfied: oss << "AlreadySatisfied"; break;
			case core::CCameraGoalSolver::SApplyResult::EStatus::AppliedAbsoluteOnly: oss << "AppliedAbsoluteOnly"; break;
			case core::CCameraGoalSolver::SApplyResult::EStatus::AppliedVirtualEvents: oss << "AppliedVirtualEvents"; break;
			case core::CCameraGoalSolver::SApplyResult::EStatus::AppliedAbsoluteAndVirtualEvents: oss << "AppliedAbsoluteAndVirtualEvents"; break;
		}
		oss << " exact=" << (result.exact ? "true" : "false")
			<< " events=" << result.eventCount;

		if (result.issues != core::CCameraGoalSolver::SApplyResult::EIssue::NoIssue)
		{
			oss << " issues=";
			bool first = true;
			auto appendIssue = [&](const char* label, const core::CCameraGoalSolver::SApplyResult::EIssue issue) -> void
			{
				if (!result.hasIssue(issue))
					return;
				if (!first)
					oss << ",";
				oss << label;
				first = false;
			};

			appendIssue("absolute_pose_fallback", core::CCameraGoalSolver::SApplyResult::EIssue::UsedAbsolutePoseFallback);
			appendIssue("missing_spherical_state", core::CCameraGoalSolver::SApplyResult::EIssue::MissingSphericalTargetState);
			appendIssue("missing_path_state", core::CCameraGoalSolver::SApplyResult::EIssue::MissingPathState);
			appendIssue("missing_dynamic_perspective_state", core::CCameraGoalSolver::SApplyResult::EIssue::MissingDynamicPerspectiveState);
			appendIssue("virtual_event_replay_failed", core::CCameraGoalSolver::SApplyResult::EIssue::VirtualEventReplayFailed);
		}

		return oss.str();
	}

	/// @brief Describe compatibility preview for applying one analyzed goal to a target camera.
	static inline std::string describeGoalApplyCompatibility(const core::SCameraGoalApplyAnalysis& analysis, const core::ICamera* targetCamera)
	{
		if (!analysis.hasCamera)
			return "No active camera";

		std::ostringstream oss;
		oss << (analysis.compatibility.exact ? "Exact" : "Best-effort")
			<< " | source=" << getCameraTypeLabel(analysis.goal.sourceKind)
			<< " | target=" << getCameraTypeLabel(targetCamera);

		if (analysis.compatibility.missingGoalStateMask != core::ICamera::GoalStateNone)
			oss << " | missing=" << describeGoalStateMask(analysis.compatibility.missingGoalStateMask);
		else if (!analysis.compatibility.sameKind && analysis.goal.sourceKind != core::ICamera::CameraKind::Unknown)
			oss << " | shared goal state only";

		return oss.str();
	}

	/// @brief Describe whether an analyzed goal can be meaningfully applied to the target camera.
	static inline std::string describeGoalApplyPolicy(const core::SCameraGoalApplyAnalysis& analysis)
	{
		if (!analysis.hasCamera)
			return "Blocked | no active camera";
		if (!analysis.finiteGoal)
			return "Blocked | invalid goal state";

		std::ostringstream oss;
		oss << (analysis.compatibility.exact ? "Exact apply" : "Best-effort apply");
		if (analysis.compatibility.missingGoalStateMask != core::ICamera::GoalStateNone)
			oss << " | drops=" << describeGoalStateMask(analysis.compatibility.missingGoalStateMask);
		else if (!analysis.compatibility.sameKind && analysis.goal.sourceKind != core::ICamera::CameraKind::Unknown)
			oss << " | shared goal state only";
		else
			oss << " | full preview available";

		return oss.str();
	}

	/// @brief Describe whether one analyzed camera state can be captured into a reusable goal.
	static inline std::string describeCameraCapturePolicy(const core::SCameraCaptureAnalysis& analysis, const core::ICamera* camera)
	{
		if (!analysis.hasCamera)
			return "Blocked | no active camera";
		if (!analysis.capturedGoal)
			return "Blocked | goal capture failed";
		if (!analysis.finiteGoal)
			return "Blocked | invalid goal state";

		std::ostringstream oss;
		oss << "Ready | source=" << getCameraTypeLabel(camera)
			<< " | goal=" << describeGoalStateMask(analysis.goal.sourceGoalStateMask);
		return oss.str();
	}

	/// @brief Describe the aggregate outcome of applying one preset to multiple cameras.
	static inline std::string describePresetApplySummary(const core::SCameraPresetApplySummary& summary, std::string_view noTargetsLabel, std::string_view prefix = "Playback apply")
	{
		if (!summary.hasTargets())
			return std::string(noTargetsLabel);

		std::ostringstream oss;
		oss << prefix << " | targets=" << summary.targetCount << " | ok=" << summary.successCount;
		if (summary.approximateCount > 0u)
			oss << " | approximate=" << summary.approximateCount;
		if (summary.failureCount > 0u)
			oss << " | failed=" << summary.failureCount;
		return oss.str();
	}
};

} // namespace nbl::ui

#endif // _C_CAMERA_TEXT_UTILITIES_HPP_
