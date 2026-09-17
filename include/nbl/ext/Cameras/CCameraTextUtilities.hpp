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

namespace nbl::ext::cameras
{

struct CCameraTextUtilities final
{
public:
	/// @brief Return a short human-readable label for a camera kind.
	static inline std::string_view getCameraTypeLabel(const ICamera::CameraKind kind)
	{
		return CCameraKindUtilities::getCameraKindLabel(kind);
	}

	/// @brief Return a short human-readable label for a concrete camera instance.
	static inline std::string_view getCameraTypeLabel(const ICamera* camera)
	{
		return camera ? getCameraTypeLabel(camera->getKind()) : "Unknown";
	}

	/// @brief Return a short human-readable description for a camera kind.
	static inline std::string_view getCameraTypeDescription(const ICamera::CameraKind kind)
	{
		return CCameraKindUtilities::getCameraKindDescription(kind);
	}

	/// @brief Return a short human-readable description for a concrete camera instance.
	static inline std::string_view getCameraTypeDescription(const ICamera* camera)
	{
		return camera ? getCameraTypeDescription(camera->getKind()) : "Unspecified camera behavior";
	}

	/// @brief Return a short human-readable label for a follow mode.
	static inline constexpr const char* getCameraFollowModeLabel(const ECameraFollowMode mode)
	{
		switch (mode)
		{
			case ECameraFollowMode::Disabled: return "Disabled";
			case ECameraFollowMode::OrbitTarget: return "Orbit target";
			case ECameraFollowMode::LookAtTarget: return "Look at target";
			case ECameraFollowMode::KeepWorldOffset: return "Keep world offset";
			case ECameraFollowMode::KeepLocalOffset: return "Keep local offset";
			default: return "Unknown";
		}
	}

	/// @brief Return a short human-readable description for a follow mode.
	static inline constexpr const char* getCameraFollowModeDescription(const ECameraFollowMode mode)
	{
		switch (mode)
		{
			case ECameraFollowMode::Disabled: return "Follow disabled";
			case ECameraFollowMode::OrbitTarget: return "Keep orbit around moving target and keep it centered";
			case ECameraFollowMode::LookAtTarget: return "Keep camera position and lock the view onto the target";
			case ECameraFollowMode::KeepWorldOffset: return "Move with the target in world offset and keep it centered";
			case ECameraFollowMode::KeepLocalOffset: return "Move with the target in target-local offset and keep it centered";
			default: return "Unknown follow mode";
		}
	}

	/// @brief Describe the typed goal-state mask in a stable human-readable format.
	static inline std::string describeGoalStateMask(const ICamera::goal_state_flags_t mask)
	{
		if (mask == ICamera::GoalStateNone)
			return "Pose only";

		std::string out;
		auto append = [&](const char* label, const ICamera::GoalStateMask bit) -> void
		{
			if (!mask.hasFlags(bit))
				return;
			if (!out.empty())
				out += ", ";
			out += label;
		};

		append("Spherical target", ICamera::GoalStateSphericalTarget);
		append("Dynamic perspective", ICamera::GoalStateDynamicPerspective);
		append("Path rig state", ICamera::GoalStatePath);
		return out;
	}

	/// @brief Describe a detailed goal-apply result for logs, smoke tests, and UI summaries.
	static inline std::string describeApplyResult(const CCameraGoalSolver::SApplyResult& result)
	{
		std::ostringstream oss;
		oss << "status=";
		switch (result.status)
		{
			case CCameraGoalSolver::SApplyResult::EStatus::Unsupported: oss << "Unsupported"; break;
			case CCameraGoalSolver::SApplyResult::EStatus::Failed: oss << "Failed"; break;
			case CCameraGoalSolver::SApplyResult::EStatus::AlreadySatisfied: oss << "AlreadySatisfied"; break;
			case CCameraGoalSolver::SApplyResult::EStatus::AppliedAbsoluteOnly: oss << "AppliedAbsoluteOnly"; break;
			case CCameraGoalSolver::SApplyResult::EStatus::AppliedVirtualEvents: oss << "AppliedVirtualEvents"; break;
			case CCameraGoalSolver::SApplyResult::EStatus::AppliedAbsoluteAndVirtualEvents: oss << "AppliedAbsoluteAndVirtualEvents"; break;
		}
		oss << " exact=" << (result.exact ? "true" : "false")
			<< " events=" << result.eventCount;

		if (result.issues != CCameraGoalSolver::SApplyResult::EIssue::NoIssue)
		{
			oss << " issues=";
			bool first = true;
			auto appendIssue = [&](const char* label, const CCameraGoalSolver::SApplyResult::EIssue issue) -> void
			{
				if (!result.hasIssue(issue))
					return;
				if (!first)
					oss << ",";
				oss << label;
				first = false;
			};

			appendIssue("absolute_pose_fallback", CCameraGoalSolver::SApplyResult::EIssue::UsedAbsolutePoseFallback);
			appendIssue("missing_spherical_state", CCameraGoalSolver::SApplyResult::EIssue::MissingSphericalTargetState);
			appendIssue("missing_path_state", CCameraGoalSolver::SApplyResult::EIssue::MissingPathState);
			appendIssue("missing_dynamic_perspective_state", CCameraGoalSolver::SApplyResult::EIssue::MissingDynamicPerspectiveState);
			appendIssue("virtual_event_replay_failed", CCameraGoalSolver::SApplyResult::EIssue::VirtualEventReplayFailed);
		}

		return oss.str();
	}

	/// @brief Describe compatibility preview for applying one analyzed goal to a target camera.
	static inline std::string describeGoalApplyCompatibility(const SCameraGoalApplyAnalysis& analysis, const ICamera* targetCamera)
	{
		if (!analysis.hasCamera)
			return "No active camera";

		std::ostringstream oss;
		oss << (analysis.compatibility.exact ? "Exact" : "Best-effort")
			<< " | source=" << getCameraTypeLabel(analysis.goal.sourceKind)
			<< " | target=" << getCameraTypeLabel(targetCamera);

		if (analysis.compatibility.missingGoalStateMask != ICamera::GoalStateNone)
			oss << " | missing=" << describeGoalStateMask(analysis.compatibility.missingGoalStateMask);
		else if (!analysis.compatibility.sameKind && analysis.goal.sourceKind != ICamera::CameraKind::Unknown)
			oss << " | shared goal state only";

		return oss.str();
	}

	/// @brief Describe whether an analyzed goal can be meaningfully applied to the target camera.
	static inline std::string describeGoalApplyPolicy(const SCameraGoalApplyAnalysis& analysis)
	{
		if (!analysis.hasCamera)
			return "Blocked | no active camera";
		if (!analysis.finiteGoal)
			return "Blocked | invalid goal state";

		std::ostringstream oss;
		oss << (analysis.compatibility.exact ? "Exact apply" : "Best-effort apply");
		if (analysis.compatibility.missingGoalStateMask != ICamera::GoalStateNone)
			oss << " | drops=" << describeGoalStateMask(analysis.compatibility.missingGoalStateMask);
		else if (!analysis.compatibility.sameKind && analysis.goal.sourceKind != ICamera::CameraKind::Unknown)
			oss << " | shared goal state only";
		else
			oss << " | full preview available";

		return oss.str();
	}

	/// @brief Describe whether one analyzed camera state can be captured into a reusable goal.
	static inline std::string describeCameraCapturePolicy(const SCameraCaptureAnalysis& analysis, const ICamera* camera)
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
	static inline std::string describePresetApplySummary(const SCameraPresetApplySummary& summary, std::string_view noTargetsLabel, std::string_view prefix = "Playback apply")
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

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_TEXT_UTILITIES_HPP_
