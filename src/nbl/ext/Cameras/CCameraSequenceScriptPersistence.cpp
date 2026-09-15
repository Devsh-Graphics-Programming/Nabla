// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraSequenceScriptPersistence.hpp"

#include <algorithm>
#include <array>
#include <string_view>
#include <type_traits>

#include "CCameraJsonPersistenceUtilities.hpp"
#include "nbl/ext/Cameras/CCameraFileUtilities.hpp"
#include "nlohmann/json.hpp"

using json_t = nlohmann::json;

namespace nbl::system
{

namespace impl
{

struct CCameraSequenceScriptJsonUtilities final
{
static bool tryParseCaptureFractionJson(const json_t& entry, float& outFraction)
{
    if (entry.is_number())
    {
        outFraction = std::clamp(entry.get<float>(), 0.f, 1.f);
        return true;
    }

    if (!entry.is_string())
        return false;

    const auto tag = entry.get<std::string>();
    if (tag == "start")
        outFraction = 0.f;
    else if (tag == "mid" || tag == "middle")
        outFraction = 0.5f;
    else if (tag == "end")
        outFraction = 1.f;
    else
        return false;

    return true;
}

template<typename T>
static void readVector3(const json_t& entry, T& outValue)
{
    using scalar_t = std::remove_reference_t<decltype(outValue.x)>;
    const auto values = entry.get<std::array<scalar_t, 3>>();
    outValue = T(values[0], values[1], values[2]);
}

static bool deserializeSequencePresentationsJson(const json_t& root, std::vector<nbl::core::CCameraSequencePresentation>& out, std::string* error)
{
    out.clear();
    if (!root.is_array())
    {
        if (error)
            *error = "Sequence presentations must be an array.";
        return false;
    }

    for (const auto& entry : root)
    {
        if (!entry.is_object() || !entry.contains("projection"))
        {
            if (error)
                *error = "Sequence presentation entry missing \"projection\".";
            return false;
        }

        nbl::core::CCameraSequencePresentation presentation;
        if (!nbl::core::CCameraSequenceScriptUtilities::tryParseProjectionType(entry["projection"].get<std::string>(), presentation.projection))
        {
            if (error)
                *error = "Sequence presentation has invalid projection type.";
            return false;
        }
        if (entry.contains("left_handed"))
            presentation.leftHanded = entry["left_handed"].get<bool>();
        out.emplace_back(presentation);
    }

    return true;
}

static bool deserializeSequenceContinuityJson(const json_t& root, nbl::core::CCameraSequenceContinuitySettings& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence continuity settings must be an object.";
        return false;
    }

    out = {};
    if (root.contains("baseline"))
        out.baseline = root["baseline"].get<bool>();
    if (root.contains("step"))
        out.step = root["step"].get<bool>();

    if (root.contains("min_pos_delta"))
    {
        out.minPosDelta = root["min_pos_delta"].get<float>();
        out.hasPosDeltaConstraint = true;
    }
    if (root.contains("max_pos_delta"))
    {
        out.maxPosDelta = root["max_pos_delta"].get<float>();
        out.hasPosDeltaConstraint = true;
    }
    else if (root.contains("pos_tolerance"))
    {
        out.maxPosDelta = root["pos_tolerance"].get<float>();
        out.hasPosDeltaConstraint = true;
    }

    if (root.contains("min_euler_delta_deg"))
    {
        out.minEulerDeltaDeg = root["min_euler_delta_deg"].get<float>();
        out.hasEulerDeltaConstraint = true;
    }
    if (root.contains("max_euler_delta_deg"))
    {
        out.maxEulerDeltaDeg = root["max_euler_delta_deg"].get<float>();
        out.hasEulerDeltaConstraint = true;
    }
    else if (root.contains("euler_tolerance_deg"))
    {
        out.maxEulerDeltaDeg = root["euler_tolerance_deg"].get<float>();
        out.hasEulerDeltaConstraint = true;
    }

    if (root.contains("disable_pos_delta"))
        out.hasPosDeltaConstraint = !root["disable_pos_delta"].get<bool>();
    if (root.contains("disable_euler_delta"))
        out.hasEulerDeltaConstraint = !root["disable_euler_delta"].get<bool>();

    if (out.step && !(out.hasPosDeltaConstraint || out.hasEulerDeltaConstraint))
    {
        if (error)
            *error = "Sequence continuity step checks require at least one delta constraint.";
        return false;
    }

    return true;
}

static bool deserializeSequenceGoalDeltaJson(const json_t& root, nbl::core::CCameraSequenceGoalDelta& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence keyframe delta must be an object.";
        return false;
    }

    out = {};
    if (root.contains("position_offset"))
    {
        readVector3(root["position_offset"], out.positionOffset);
        out.hasPositionOffset = true;
    }
    if (root.contains("rotation_euler_deg_offset"))
    {
        readVector3(root["rotation_euler_deg_offset"], out.rotationEulerDegOffset);
        out.hasRotationEulerDegOffset = true;
    }
    if (root.contains("target_offset"))
    {
        readVector3(root["target_offset"], out.targetOffset);
        out.hasTargetOffset = true;
    }
    if (root.contains("orbit_u_delta_deg"))
        out.orbitDelta.setUDeltaDeg(root["orbit_u_delta_deg"].get<double>());
    if (root.contains("orbit_v_delta_deg"))
        out.orbitDelta.setVDeltaDeg(root["orbit_v_delta_deg"].get<double>());
    if (root.contains("orbit_distance_delta"))
        out.orbitDelta.setDistanceDelta(root["orbit_distance_delta"].get<float>());
    if (root.contains("path_s_delta_deg"))
        out.pathDelta.setSDeltaDeg(root["path_s_delta_deg"].get<double>());
    if (root.contains("path_u_delta"))
        out.pathDelta.setUDelta(root["path_u_delta"].get<double>());
    if (root.contains("path_v_delta"))
        out.pathDelta.setVDelta(root["path_v_delta"].get<double>());
    if (root.contains("path_roll_delta_deg"))
        out.pathDelta.setRollDeltaDeg(root["path_roll_delta_deg"].get<double>());
    if (root.contains("dynamic_base_fov_delta"))
    {
        out.dynamicBaseFovDelta = root["dynamic_base_fov_delta"].get<float>();
        out.hasDynamicBaseFovDelta = true;
    }
    if (root.contains("dynamic_reference_distance_delta"))
    {
        out.dynamicReferenceDistanceDelta = root["dynamic_reference_distance_delta"].get<float>();
        out.hasDynamicReferenceDistanceDelta = true;
    }

    return true;
}

static bool deserializeSequenceKeyframeJson(const json_t& root, nbl::core::CCameraSequenceKeyframe& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence keyframe must be an object.";
        return false;
    }

    out = {};
    if (root.contains("time"))
        out.time = std::max(0.f, root["time"].get<float>());

    if (root.contains("delta"))
    {
        if (!deserializeSequenceGoalDeltaJson(root["delta"], out.delta, error))
            return false;
        out.hasDelta = true;
    }

    if (root.contains("preset"))
    {
        impl::CCameraJsonPersistenceUtilities::deserializePresetJson(root["preset"], out.absolutePreset);
        out.hasAbsolutePreset = true;
    }
    else if (root.contains("position") || root.contains("orientation") || root.contains("target_position") ||
        root.contains("distance") || root.contains("orbit_u") || root.contains("orbit_v") ||
        root.contains("orbit_distance") || root.contains("path_s") || root.contains("path_u") ||
        root.contains("path_v") || root.contains("path_roll") ||
        root.contains("dynamic_base_fov") || root.contains("dynamic_reference_distance"))
    {
        impl::CCameraJsonPersistenceUtilities::deserializePresetJson(root, out.absolutePreset);
        out.hasAbsolutePreset = true;
    }

    return true;
}

static bool deserializeSequenceTrackedTargetDeltaJson(const json_t& root, nbl::core::CCameraSequenceTrackedTargetDelta& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence target delta must be an object.";
        return false;
    }

    out = {};
    if (root.contains("position_offset"))
    {
        readVector3(root["position_offset"], out.positionOffset);
        out.hasPositionOffset = true;
    }
    if (root.contains("rotation_euler_deg_offset"))
    {
        readVector3(root["rotation_euler_deg_offset"], out.rotationEulerDegOffset);
        out.hasRotationEulerDegOffset = true;
    }

    return true;
}

static bool deserializeSequenceTrackedTargetKeyframeJson(const json_t& root, nbl::core::CCameraSequenceTrackedTargetKeyframe& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence target keyframe must be an object.";
        return false;
    }

    out = {};
    if (root.contains("time"))
        out.time = std::max(0.f, root["time"].get<float>());

    if (root.contains("delta"))
    {
        if (!deserializeSequenceTrackedTargetDeltaJson(root["delta"], out.delta, error))
            return false;
        out.hasDelta = true;
    }

    if (root.contains("position"))
    {
        readVector3(root["position"], out.absolutePosition);
        out.hasAbsolutePosition = true;
    }
    if (root.contains("rotation_euler_deg"))
    {
        readVector3(root["rotation_euler_deg"], out.absoluteRotationEulerDeg);
        out.hasAbsoluteRotationEulerDeg = true;
    }

    return true;
}

static bool deserializeSequenceSegmentJson(const json_t& root, nbl::core::CCameraSequenceSegment& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence segment must be an object.";
        return false;
    }

    out = {};
    if (root.contains("name"))
        out.name = root["name"].get<std::string>();
    if (root.contains("camera_identifier"))
        out.cameraIdentifier = root["camera_identifier"].get<std::string>();
    if (root.contains("camera_kind"))
    {
        if (!nbl::core::CCameraSequenceScriptUtilities::tryParseCameraKind(root["camera_kind"].get<std::string>(), out.cameraKind))
        {
            if (error)
                *error = "Sequence segment has invalid camera_kind.";
            return false;
        }
    }
    if (root.contains("duration_seconds"))
    {
        out.durationSeconds = std::max(0.f, root["duration_seconds"].get<float>());
        out.hasDurationSeconds = true;
    }
    if (root.contains("reset_camera"))
    {
        out.resetCamera = root["reset_camera"].get<bool>();
        out.hasResetCamera = true;
    }
    if (root.contains("presentations"))
    {
        if (!deserializeSequencePresentationsJson(root["presentations"], out.presentations, error))
            return false;
    }
    if (root.contains("continuity"))
    {
        if (!deserializeSequenceContinuityJson(root["continuity"], out.continuity, error))
            return false;
        out.hasContinuity = true;
    }
    if (root.contains("captures"))
    {
        if (!root["captures"].is_array())
        {
            if (error)
                *error = "Sequence segment captures must be an array.";
            return false;
        }

        out.captureFractions.clear();
        for (const auto& entry : root["captures"])
        {
            float fraction = 0.f;
            if (!tryParseCaptureFractionJson(entry, fraction))
            {
                if (error)
                    *error = "Sequence segment capture entry is invalid.";
                return false;
            }
            out.captureFractions.emplace_back(fraction);
        }
        nbl::core::CCameraSequenceScriptUtilities::normalizeCaptureFractions(out.captureFractions);
        out.hasCaptureFractions = true;
    }
    if (root.contains("keyframes"))
    {
        if (!root["keyframes"].is_array())
        {
            if (error)
                *error = "Sequence segment keyframes must be an array.";
            return false;
        }
        for (const auto& entry : root["keyframes"])
        {
            nbl::core::CCameraSequenceKeyframe keyframe;
            if (!deserializeSequenceKeyframeJson(entry, keyframe, error))
                return false;
            out.keyframes.emplace_back(std::move(keyframe));
        }
    }
    if (root.contains("target_keyframes"))
    {
        if (!root["target_keyframes"].is_array())
        {
            if (error)
                *error = "Sequence segment target_keyframes must be an array.";
            return false;
        }
        for (const auto& entry : root["target_keyframes"])
        {
            nbl::core::CCameraSequenceTrackedTargetKeyframe keyframe;
            if (!deserializeSequenceTrackedTargetKeyframeJson(entry, keyframe, error))
                return false;
            out.targetKeyframes.emplace_back(std::move(keyframe));
        }
    }

    if (out.keyframes.empty())
    {
        if (error)
            *error = "Sequence segment requires at least one keyframe.";
        return false;
    }
    if (out.cameraKind == nbl::core::ICamera::CameraKind::Unknown && out.cameraIdentifier.empty())
    {
        if (error)
            *error = "Sequence segment requires camera_kind or camera_identifier.";
        return false;
    }

    return true;
}

static bool deserializeCameraSequenceScriptJson(const json_t& root, nbl::core::CCameraSequenceScript& out, std::string* error)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Camera sequence script must be an object.";
        return false;
    }

    out = {};
    if (root.contains("enabled"))
        out.enabled = root["enabled"].get<bool>();
    if (root.contains("log"))
        out.log = root["log"].get<bool>();
    if (root.contains("exclusive"))
        out.exclusive = root["exclusive"].get<bool>();
    if (root.contains("exclusive_input"))
        out.exclusive = root["exclusive_input"].get<bool>() || out.exclusive;
    if (root.contains("hard_fail"))
        out.hardFail = root["hard_fail"].get<bool>();
    if (root.contains("visual_debug"))
        out.visualDebug = root["visual_debug"].get<bool>();
    if (root.contains("visual_debug_target_fps"))
        out.visualDebugTargetFps = root["visual_debug_target_fps"].get<float>();
    if (root.contains("visual_debug_hold_seconds"))
        out.visualDebugHoldSeconds = root["visual_debug_hold_seconds"].get<float>();
    if (root.contains("enableActiveCameraMovement"))
    {
        out.enableActiveCameraMovement = root["enableActiveCameraMovement"].get<bool>();
        out.hasEnableActiveCameraMovement = true;
    }
    if (root.contains("capture_prefix"))
        out.capturePrefix = root["capture_prefix"].get<std::string>();
    if (root.contains("fps"))
        out.fps = std::max(1.f, root["fps"].get<float>());

    if (root.contains("defaults"))
    {
        const auto& defaults = root["defaults"];
        if (!defaults.is_object())
        {
            if (error)
                *error = "Camera sequence defaults must be an object.";
            return false;
        }

        if (defaults.contains("duration_seconds"))
            out.defaults.durationSeconds = std::max(0.f, defaults["duration_seconds"].get<float>());
        if (defaults.contains("reset_camera"))
            out.defaults.resetCamera = defaults["reset_camera"].get<bool>();
        if (defaults.contains("presentations"))
        {
            if (!deserializeSequencePresentationsJson(defaults["presentations"], out.defaults.presentations, error))
                return false;
        }
        if (defaults.contains("continuity"))
        {
            if (!deserializeSequenceContinuityJson(defaults["continuity"], out.defaults.continuity, error))
                return false;
        }
        if (defaults.contains("captures"))
        {
            if (!defaults["captures"].is_array())
            {
                if (error)
                    *error = "Camera sequence default captures must be an array.";
                return false;
            }

            out.defaults.captureFractions.clear();
            for (const auto& entry : defaults["captures"])
            {
                float fraction = 0.f;
                if (!tryParseCaptureFractionJson(entry, fraction))
                {
                    if (error)
                        *error = "Camera sequence default capture entry is invalid.";
                    return false;
                }
                out.defaults.captureFractions.emplace_back(fraction);
            }
            nbl::core::CCameraSequenceScriptUtilities::normalizeCaptureFractions(out.defaults.captureFractions);
        }
    }

    if (!root.contains("segments") || !root["segments"].is_array())
    {
        if (error)
            *error = "Camera sequence script requires a \"segments\" array.";
        return false;
    }

    for (const auto& entry : root["segments"])
    {
        nbl::core::CCameraSequenceSegment segment;
        if (!deserializeSequenceSegmentJson(entry, segment, error))
            return false;
        out.segments.emplace_back(std::move(segment));
    }

    if (out.segments.empty())
    {
        if (error)
            *error = "Camera sequence script must contain at least one segment.";
        return false;
    }

    return true;
}

}; // struct CCameraSequenceScriptJsonUtilities

} // namespace impl

bool CCameraSequenceScriptPersistenceUtilities::deserializeCameraSequenceScript(std::string_view text, core::CCameraSequenceScript& out, std::string* error)
{
    try
    {
        const auto root = json_t::parse(text);
        return impl::CCameraSequenceScriptJsonUtilities::deserializeCameraSequenceScriptJson(root, out, error);
    }
    catch (const json_t::exception& e)
    {
        if (error)
            *error = e.what();
        return false;
    }
}

bool CCameraSequenceScriptPersistenceUtilities::loadCameraSequenceScriptFromFile(ISystem& system, const path& filePath, core::CCameraSequenceScript& out, std::string* error)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text, error, "Cannot open camera sequence script file."))
        return false;

    return deserializeCameraSequenceScript(text, out, error);
}

} // namespace nbl::system
