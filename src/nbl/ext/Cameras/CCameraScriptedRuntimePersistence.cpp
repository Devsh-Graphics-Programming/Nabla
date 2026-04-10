// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraScriptedRuntimePersistence.hpp"

#include <algorithm>
#include <array>
#include <sstream>
#include <string_view>
#include <type_traits>

#include "CCameraJsonPersistenceUtilities.hpp"
#include "nlohmann/json.hpp"

using json_t = nlohmann::json;

bool tryParseCaptureFractionJson(const json_t& entry, float& outFraction)
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
void readVector3(const json_t& entry, T& outValue)
{
    using scalar_t = std::remove_reference_t<decltype(outValue.x)>;
    const auto values = entry.get<std::array<scalar_t, 3>>();
    outValue = T(values[0], values[1], values[2]);
}

bool deserializeSequencePresentationsJson(const json_t& root, std::vector<nbl::core::CCameraSequencePresentation>& out, std::string* error)
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

bool deserializeSequenceContinuityJson(const json_t& root, nbl::core::CCameraSequenceContinuitySettings& out, std::string* error)
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

bool deserializeSequenceGoalDeltaJson(const json_t& root, nbl::core::CCameraSequenceGoalDelta& out, std::string* error)
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
    {
        out.orbitDelta.setUDeltaDeg(root["orbit_u_delta_deg"].get<double>());
    }
    if (root.contains("orbit_v_delta_deg"))
    {
        out.orbitDelta.setVDeltaDeg(root["orbit_v_delta_deg"].get<double>());
    }
    if (root.contains("orbit_distance_delta"))
    {
        out.orbitDelta.setDistanceDelta(root["orbit_distance_delta"].get<float>());
    }
    if (root.contains("path_s_delta_deg"))
    {
        out.pathDelta.setSDeltaDeg(root["path_s_delta_deg"].get<double>());
    }
    if (root.contains("path_u_delta"))
    {
        out.pathDelta.setUDelta(root["path_u_delta"].get<double>());
    }
    if (root.contains("path_v_delta"))
    {
        out.pathDelta.setVDelta(root["path_v_delta"].get<double>());
    }
    if (root.contains("path_roll_delta_deg"))
    {
        out.pathDelta.setRollDeltaDeg(root["path_roll_delta_deg"].get<double>());
    }
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

bool deserializeSequenceKeyframeJson(const json_t& root, nbl::core::CCameraSequenceKeyframe& out, std::string* error)
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
        nbl::system::deserializePresetJson(root["preset"], out.absolutePreset);
        out.hasAbsolutePreset = true;
    }
    else if (root.contains("position") || root.contains("orientation") || root.contains("target_position") ||
        root.contains("distance") || root.contains("orbit_u") || root.contains("orbit_v") ||
        root.contains("orbit_distance") || root.contains("path_s") || root.contains("path_u") ||
        root.contains("path_v") || root.contains("path_roll") ||
        root.contains("dynamic_base_fov") || root.contains("dynamic_reference_distance"))
    {
        nbl::system::deserializePresetJson(root, out.absolutePreset);
        out.hasAbsolutePreset = true;
    }

    return true;
}

bool deserializeSequenceTrackedTargetDeltaJson(const json_t& root, nbl::core::CCameraSequenceTrackedTargetDelta& out, std::string* error)
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

bool deserializeSequenceTrackedTargetKeyframeJson(const json_t& root, nbl::core::CCameraSequenceTrackedTargetKeyframe& out, std::string* error)
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

bool deserializeSequenceSegmentJson(const json_t& root, nbl::core::CCameraSequenceSegment& out, std::string* error)
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

bool deserializeCameraSequenceScriptJson(const json_t& root, nbl::core::CCameraSequenceScript& out, std::string* error)
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

nbl::hlsl::float32_t4x4 composeScriptedImguizmoTransform(
    const std::array<float, 3>& translation,
    const std::array<float, 3>& rotationDeg,
    const std::array<float, 3>& scale)
{
    return nbl::hlsl::CCameraMathUtilities::composeTransformMatrix(
        nbl::hlsl::float32_t3(translation[0], translation[1], translation[2]),
        nbl::hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegrees(nbl::hlsl::float32_t3(rotationDeg[0], rotationDeg[1], rotationDeg[2])),
        nbl::hlsl::float32_t3(scale[0], scale[1], scale[2]));
}

nbl::hlsl::float32_t4x4 makeScriptedMatrixFromArray(const std::array<float, 16>& values)
{
    nbl::hlsl::float32_t4x4 out(1.f);
    for (uint32_t column = 0u; column < 4u; ++column)
    {
        for (uint32_t row = 0u; row < 4u; ++row)
            out[column][row] = values[column * 4u + row];
    }
    return out;
}

std::optional<nbl::system::CCameraScriptedInputEvent::KeyboardData::Action> parseScriptedKeyboardAction(std::string_view action)
{
    if (action == "pressed" || action == "press")
        return nbl::system::CCameraScriptedInputEvent::KeyboardData::Action::Pressed;
    if (action == "released" || action == "release")
        return nbl::system::CCameraScriptedInputEvent::KeyboardData::Action::Released;
    return std::nullopt;
}

nbl::ui::E_KEY_CODE parseScriptedKeyCode(std::string_view key)
{
    auto parsed = nbl::ui::stringToKeyCode(key);
    if (parsed != nbl::ui::EKC_NONE)
        return parsed;

    constexpr std::string_view KeyPrefix = "KEY_";
    constexpr std::string_view EkcPrefix = "EKC_";
    if (key.starts_with(KeyPrefix))
        parsed = nbl::ui::stringToKeyCode(key.substr(KeyPrefix.size()));
    if (parsed == nbl::ui::EKC_NONE && key.starts_with(EkcPrefix))
        parsed = nbl::ui::stringToKeyCode(key.substr(EkcPrefix.size()));
    return parsed;
}

std::optional<nbl::ui::E_MOUSE_BUTTON> parseScriptedMouseButton(std::string_view button)
{
    if (button == "LEFT_BUTTON")
        return nbl::ui::EMB_LEFT_BUTTON;
    if (button == "RIGHT_BUTTON")
        return nbl::ui::EMB_RIGHT_BUTTON;
    if (button == "MIDDLE_BUTTON")
        return nbl::ui::EMB_MIDDLE_BUTTON;
    if (button == "BUTTON_4")
        return nbl::ui::EMB_BUTTON_4;
    if (button == "BUTTON_5")
        return nbl::ui::EMB_BUTTON_5;
    return std::nullopt;
}

std::optional<nbl::system::CCameraScriptedInputEvent::MouseData::ClickAction> parseScriptedMouseClickAction(std::string_view action)
{
    if (action == "pressed" || action == "press")
        return nbl::system::CCameraScriptedInputEvent::MouseData::ClickAction::Pressed;
    if (action == "released" || action == "release")
        return nbl::system::CCameraScriptedInputEvent::MouseData::ClickAction::Released;
    return std::nullopt;
}

void parseScriptedCaptureFramesJson(const json_t& script, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!script.contains("capture_frames"))
        return;

    for (const auto& frame : script["capture_frames"])
        out.timeline.captureFrames.emplace_back(frame.get<uint64_t>());
}

void parseScriptedControlOverridesJson(const json_t& controls, nbl::system::CCameraScriptedControlOverrides& out)
{
    if (controls.contains("keyboard_scale"))
    {
        out.hasKeyboardScale = true;
        out.keyboardScale = controls["keyboard_scale"].get<float>();
    }
    if (controls.contains("mouse_move_scale"))
    {
        out.hasMouseMoveScale = true;
        out.mouseMoveScale = controls["mouse_move_scale"].get<float>();
    }
    if (controls.contains("mouse_scroll_scale"))
    {
        out.hasMouseScrollScale = true;
        out.mouseScrollScale = controls["mouse_scroll_scale"].get<float>();
    }
    if (controls.contains("translation_scale"))
    {
        out.hasTranslationScale = true;
        out.translationScale = controls["translation_scale"].get<float>();
    }
    if (controls.contains("rotation_scale"))
    {
        out.hasRotationScale = true;
        out.rotationScale = controls["rotation_scale"].get<float>();
    }
}

bool parseScriptedSequenceIfPresentJson(const json_t& script, nbl::system::CCameraScriptedInputParseResult& out, std::string* error)
{
    if (!script.contains("segments"))
        return true;

    nbl::core::CCameraSequenceScript sequence;
    if (!deserializeCameraSequenceScriptJson(script, sequence, error))
        return false;

    out.sequence = std::move(sequence);
    return true;
}

void appendScriptedCaptureFrame(nbl::system::CCameraScriptedInputParseResult& out, const uint64_t frame, const bool captureFrame)
{
    if (captureFrame)
        out.timeline.captureFrames.emplace_back(frame);
}

void parseScriptedKeyboardEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("key") || !event.contains("action"))
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted keyboard event missing \"key\" or \"action\".");
        return;
    }

    const auto keyText = event["key"].get<std::string>();
    const auto actionText = event["action"].get<std::string>();
    const auto key = parseScriptedKeyCode(keyText);
    if (key == nbl::ui::EKC_NONE)
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted keyboard event has invalid key \"" + keyText + "\".");
        return;
    }

    const auto action = parseScriptedKeyboardAction(actionText);
    if (!action.has_value())
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted keyboard event has invalid action \"" + actionText + "\".");
        return;
    }

    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Keyboard;
    entry.keyboard.key = key;
    entry.keyboard.action = action.value();
    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

void parseScriptedMouseEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("kind"))
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted mouse event missing \"kind\".");
        return;
    }

    const auto kind = event["kind"].get<std::string>();
    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Mouse;

    if (kind == "move")
    {
        entry.mouse.type = nbl::system::CCameraScriptedInputEvent::MouseData::Type::Movement;
        entry.mouse.dx = event.value("dx", 0);
        entry.mouse.dy = event.value("dy", 0);
    }
    else if (kind == "scroll")
    {
        entry.mouse.type = nbl::system::CCameraScriptedInputEvent::MouseData::Type::Scroll;
        entry.mouse.v = event.value("v", 0);
        entry.mouse.h = event.value("h", 0);
    }
    else if (kind == "click")
    {
        if (!event.contains("button") || !event.contains("action"))
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted click event missing \"button\" or \"action\".");
            return;
        }

        const auto buttonText = event["button"].get<std::string>();
        const auto actionText = event["action"].get<std::string>();
        const auto button = parseScriptedMouseButton(buttonText);
        if (!button.has_value())
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted click event has invalid button \"" + buttonText + "\".");
            return;
        }

        const auto action = parseScriptedMouseClickAction(actionText);
        if (!action.has_value())
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted click event has invalid action \"" + actionText + "\".");
            return;
        }

        entry.mouse.type = nbl::system::CCameraScriptedInputEvent::MouseData::Type::Click;
        entry.mouse.button = button.value();
        entry.mouse.action = action.value();
        entry.mouse.x = event.value("x", 0);
        entry.mouse.y = event.value("y", 0);
    }
    else
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted mouse event has invalid kind \"" + kind + "\".");
        return;
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

void parseScriptedImguizmoEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::system::CCameraScriptedInputParseResult& out)
{
    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Imguizmo;

    if (event.contains("delta_trs"))
    {
        const auto matrix = event["delta_trs"].get<std::array<float, 16>>();
        entry.imguizmo = makeScriptedMatrixFromArray(matrix);
    }
    else
    {
        const auto translation = event.contains("translation") ? event["translation"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
        const auto rotation = event.contains("rotation_deg") ? event["rotation_deg"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
        const auto scale = event.contains("scale") ? event["scale"].get<std::array<float, 3>>() : std::array<float, 3>{1.f, 1.f, 1.f};
        entry.imguizmo = composeScriptedImguizmoTransform(translation, rotation, scale);
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

int32_t parseScriptedActionIntValue(const json_t& event)
{
    if (event.contains("value"))
        return event["value"].get<int32_t>();
    if (event.contains("index"))
        return event["index"].get<int32_t>();
    return 0;
}

bool parseScriptedProjectionActionValue(const json_t& event, nbl::system::CCameraScriptedInputEvent::ActionData& action, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (event.contains("value") && event["value"].is_string())
    {
        const auto valueText = event["value"].get<std::string>();
        if (valueText == "perspective")
            action.value = static_cast<int32_t>(nbl::core::IPlanarProjection::CProjection::Perspective);
        else if (valueText == "orthographic")
            action.value = static_cast<int32_t>(nbl::core::IPlanarProjection::CProjection::Orthographic);
        else
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted action projection type has invalid value \"" + valueText + "\".");
            return false;
        }
    }
    else
    {
        action.value = parseScriptedActionIntValue(event);
    }

    return true;
}

void parseScriptedActionEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("action"))
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted action event missing \"action\".");
        return;
    }

    const auto actionText = event["action"].get<std::string>();
    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Action;

    if (actionText == "set_active_render_window")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow;
        entry.action.value = parseScriptedActionIntValue(event);
    }
    else if (actionText == "set_active_planar")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar;
        entry.action.value = parseScriptedActionIntValue(event);
    }
    else if (actionText == "set_projection_type")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType;
        if (!parseScriptedProjectionActionValue(event, entry.action, out))
            return;
    }
    else if (actionText == "set_projection_index")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::SetProjectionIndex;
        entry.action.value = parseScriptedActionIntValue(event);
    }
    else if (actionText == "set_use_window")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::SetUseWindow;
        entry.action.value = event.value("value", false) ? 1 : 0;
    }
    else if (actionText == "set_left_handed")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded;
        entry.action.value = event.value("value", false) ? 1 : 0;
    }
    else if (actionText == "reset_active_camera")
    {
        entry.action.kind = nbl::system::CCameraScriptedInputEvent::ActionData::Kind::ResetActiveCamera;
        entry.action.value = 1;
    }
    else
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted action event has invalid action \"" + actionText + "\".");
        return;
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

void parseScriptedInputEventJson(const json_t& event, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("frame") || !event.contains("type"))
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted input event missing \"frame\" or \"type\".");
        return;
    }

    const auto frame = event["frame"].get<uint64_t>();
    const auto type = event["type"].get<std::string>();
    const bool captureFrame = event.value("capture", false);

    if (type == "keyboard")
        parseScriptedKeyboardEventJson(event, frame, captureFrame, out);
    else if (type == "mouse")
        parseScriptedMouseEventJson(event, frame, captureFrame, out);
    else if (type == "imguizmo")
        parseScriptedImguizmoEventJson(event, frame, captureFrame, out);
    else if (type == "action")
        parseScriptedActionEventJson(event, frame, captureFrame, out);
    else
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted input event has invalid type \"" + type + "\".");
}

void parseScriptedInputEventsJson(const json_t& script, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!script.contains("events"))
        return;

    for (const auto& event : script["events"])
        parseScriptedInputEventJson(event, out);
}

bool parseScriptedImguizmoVirtualCheckJson(const json_t& check, nbl::system::CCameraScriptedInputCheck& outCheck, nbl::system::CCameraScriptedInputParseResult& out)
{
    outCheck.kind = nbl::system::CCameraScriptedInputCheck::Kind::ImguizmoVirtual;
    outCheck.tolerance = check.value("tolerance", outCheck.tolerance);

    if (!check.contains("events"))
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Imguizmo virtual check missing \"events\".");
        return false;
    }

    for (const auto& expectedEvent : check["events"])
    {
        if (!expectedEvent.contains("type") || !expectedEvent.contains("magnitude"))
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Imguizmo virtual check event missing \"type\" or \"magnitude\".");
            continue;
        }

        const auto typeText = expectedEvent["type"].get<std::string>();
        const auto type = nbl::core::CVirtualGimbalEvent::stringToVirtualEvent(typeText);
        if (type == nbl::core::CVirtualGimbalEvent::None)
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Imguizmo virtual check event has invalid type \"" + typeText + "\".");
            continue;
        }

        nbl::system::CCameraScriptedInputCheck::ExpectedVirtualEvent expected;
        expected.type = type;
        expected.magnitude = expectedEvent["magnitude"].get<double>();
        outCheck.expectedVirtualEvents.emplace_back(expected);
    }

    return true;
}

bool parseScriptedCheckJson(const json_t& check, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!check.contains("frame") || !check.contains("kind"))
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted check missing \"frame\" or \"kind\".");
        return false;
    }

    const auto frame = check["frame"].get<uint64_t>();
    const auto kind = check["kind"].get<std::string>();

    nbl::system::CCameraScriptedInputCheck entry;
    entry.frame = frame;

    if (kind == "baseline")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::Baseline;
    }
    else if (kind == "imguizmo_virtual")
    {
        if (!parseScriptedImguizmoVirtualCheckJson(check, entry, out))
            return false;
    }
    else if (kind == "gimbal_near")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::GimbalNear;
        entry.posTolerance = check.value("pos_tolerance", entry.posTolerance);
        entry.eulerToleranceDeg = check.value("euler_tolerance_deg", entry.eulerToleranceDeg);

        if (check.contains("position"))
        {
            readVector3(check["position"], entry.expectedPos);
            entry.hasExpectedPos = true;
        }
        if (check.contains("euler_deg"))
        {
            readVector3(check["euler_deg"], entry.expectedEulerDeg);
            entry.hasExpectedEuler = true;
        }
    }
    else if (kind == "gimbal_delta")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::GimbalDelta;
        entry.posTolerance = check.value("pos_tolerance", entry.posTolerance);
        entry.eulerToleranceDeg = check.value("euler_tolerance_deg", entry.eulerToleranceDeg);
    }
    else if (kind == "gimbal_step")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::GimbalStep;

        if (check.contains("min_pos_delta"))
        {
            entry.minPosDelta = check["min_pos_delta"].get<float>();
            entry.hasPosDeltaConstraint = true;
        }
        if (check.contains("max_pos_delta"))
        {
            entry.posTolerance = check["max_pos_delta"].get<float>();
            entry.hasPosDeltaConstraint = true;
        }
        else if (check.contains("pos_tolerance"))
        {
            entry.posTolerance = check["pos_tolerance"].get<float>();
            entry.hasPosDeltaConstraint = true;
        }

        if (check.contains("min_euler_delta_deg"))
        {
            entry.minEulerDeltaDeg = check["min_euler_delta_deg"].get<float>();
            entry.hasEulerDeltaConstraint = true;
        }
        if (check.contains("max_euler_delta_deg"))
        {
            entry.eulerToleranceDeg = check["max_euler_delta_deg"].get<float>();
            entry.hasEulerDeltaConstraint = true;
        }
        else if (check.contains("euler_tolerance_deg"))
        {
            entry.eulerToleranceDeg = check["euler_tolerance_deg"].get<float>();
            entry.hasEulerDeltaConstraint = true;
        }

        if (!entry.hasPosDeltaConstraint && !entry.hasEulerDeltaConstraint)
        {
            nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "gimbal_step check requires at least one delta constraint.");
            return false;
        }
    }
    else
    {
        nbl::system::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted check has invalid kind \"" + kind + "\".");
        return false;
    }

    out.timeline.checks.emplace_back(std::move(entry));
    return true;
}

void parseScriptedChecksJson(const json_t& script, nbl::system::CCameraScriptedInputParseResult& out)
{
    if (!script.contains("checks"))
        return;

    for (const auto& check : script["checks"])
        parseScriptedCheckJson(check, out);
}

namespace nbl::system
{

bool readCameraSequenceScript(std::istream& in, core::CCameraSequenceScript& out, std::string* error)
{
    if (!in)
    {
        if (error)
            *error = "Input stream is not readable.";
        return false;
    }

    json_t root;
    in >> root;
    return deserializeCameraSequenceScriptJson(root, out, error);
}

bool readCameraSequenceScript(std::string_view text, core::CCameraSequenceScript& out, std::string* error)
{
    std::istringstream stream{std::string(text)};
    return readCameraSequenceScript(stream, out, error);
}

bool loadCameraSequenceScriptFromFile(ISystem& system, const path& filePath, core::CCameraSequenceScript& out, std::string* error)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text, error, "Cannot open camera sequence script file."))
        return false;

    return readCameraSequenceScript(text, out, error);
}

bool readCameraScriptedInput(std::istream& in, CCameraScriptedInputParseResult& out, std::string* error)
{
    if (!in)
    {
        if (error)
            *error = "Input stream is not readable.";
        return false;
    }

    json_t script;
    in >> script;

    out = {};

    if (script.contains("enabled"))
        out.enabled = script["enabled"].get<bool>();
    if (script.contains("log"))
    {
        out.hasLog = true;
        out.log = script["log"].get<bool>();
    }
    if (script.contains("hard_fail"))
        out.hardFail = script["hard_fail"].get<bool>();
    if (script.contains("visual_debug"))
        out.visualDebug = script["visual_debug"].get<bool>();
    if (script.contains("visual_debug_target_fps"))
        out.visualTargetFps = script["visual_debug_target_fps"].get<float>();
    if (script.contains("visual_debug_hold_seconds"))
        out.visualCameraHoldSeconds = script["visual_debug_hold_seconds"].get<float>();
    if (script.contains("enableActiveCameraMovement"))
    {
        out.hasEnableActiveCameraMovement = true;
        out.enableActiveCameraMovement = script["enableActiveCameraMovement"].get<bool>();
    }
    if (script.contains("exclusive_input"))
        out.exclusive = script["exclusive_input"].get<bool>() || out.exclusive;
    if (script.contains("exclusive"))
        out.exclusive = script["exclusive"].get<bool>() || out.exclusive;
    if (script.contains("capture_prefix"))
        out.capturePrefix = script["capture_prefix"].get<std::string>();
    if (out.capturePrefix.empty())
        out.capturePrefix = "script";

    parseScriptedCaptureFramesJson(script, out);

    if (script.contains("camera_controls"))
        parseScriptedControlOverridesJson(script["camera_controls"], out.cameraControls);

    if (!parseScriptedSequenceIfPresentJson(script, out, error))
        return false;

    parseScriptedInputEventsJson(script, out);
    parseScriptedChecksJson(script, out);

    CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(out.timeline);
    return true;
}

bool readCameraScriptedInput(std::string_view text, CCameraScriptedInputParseResult& out, std::string* error)
{
    std::istringstream stream{std::string(text)};
    return readCameraScriptedInput(stream, out, error);
}

bool loadCameraScriptedInputFromFile(ISystem& system, const path& filePath, CCameraScriptedInputParseResult& out, std::string* error)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text, error, "Cannot open scripted input file."))
        return false;

    return readCameraScriptedInput(text, out, error);
}

} // namespace nbl::system
