// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraPersistence.hpp"

#include <array>
#include <sstream>

#include "CCameraJsonPersistenceUtilities.hpp"
#include "nlohmann/json.hpp"

using json_t = nlohmann::json;

json_t serializeGoalJson(const nbl::core::CCameraGoal& goal)
{
    json_t json;
    json["position"] = { goal.position.x, goal.position.y, goal.position.z };
    json["orientation"] = {
        goal.orientation.data.x,
        goal.orientation.data.y,
        goal.orientation.data.z,
        goal.orientation.data.w
    };
    json["camera_kind"] = static_cast<uint32_t>(goal.sourceKind);
    json["camera_capabilities"] = goal.sourceCapabilities;
    json["camera_goal_state_mask"] = goal.sourceGoalStateMask;

    if (goal.hasTargetPosition)
        json["target_position"] = { goal.targetPosition.x, goal.targetPosition.y, goal.targetPosition.z };
    if (goal.hasDistance)
        json["distance"] = goal.distance;
    if (goal.hasOrbitState)
    {
        json["orbit_u"] = goal.orbitUv.x;
        json["orbit_v"] = goal.orbitUv.y;
        json["orbit_distance"] = goal.orbitDistance;
    }
    if (goal.hasPathState)
    {
        json["path_s"] = goal.pathState.s;
        json["path_u"] = goal.pathState.u;
        json["path_v"] = goal.pathState.v;
        json["path_roll"] = goal.pathState.roll;
    }
    if (goal.hasDynamicPerspectiveState)
    {
        json["dynamic_base_fov"] = goal.dynamicPerspectiveState.baseFov;
        json["dynamic_reference_distance"] = goal.dynamicPerspectiveState.referenceDistance;
    }

    return json;
}

json_t serializePresetJson(const nbl::core::CCameraPreset& preset)
{
    auto json = serializeGoalJson(nbl::core::CCameraPresetUtilities::makeGoalFromPreset(preset));
    json["name"] = preset.name;
    json["identifier"] = preset.identifier;
    return json;
}

json_t serializeKeyframeTrackJson(const nbl::core::CCameraKeyframeTrack& track)
{
    json_t root;
    root["keyframes"] = json_t::array();

    for (const auto& keyframe : track.keyframes)
    {
        auto json = serializePresetJson(keyframe.preset);
        json["time"] = keyframe.time;
        root["keyframes"].push_back(std::move(json));
    }

    return root;
}

bool deserializeKeyframeTrackJson(const json_t& root, nbl::core::CCameraKeyframeTrack& track)
{
    if (!root.contains("keyframes") || !root["keyframes"].is_array())
        return false;

    track = {};
    for (const auto& entry : root["keyframes"])
    {
        nbl::core::CCameraKeyframe keyframe;
        if (entry.contains("time"))
            keyframe.time = std::max(0.f, entry["time"].get<float>());
        nbl::system::deserializePresetJson(entry, keyframe.preset);
        track.keyframes.emplace_back(std::move(keyframe));
    }

    nbl::core::CCameraKeyframeTrackUtilities::sortKeyframeTrackByTime(track);
    nbl::core::CCameraKeyframeTrackUtilities::normalizeSelectedKeyframeTrack(track);
    return true;
}

json_t serializePresetCollectionJson(std::span<const nbl::core::CCameraPreset> presets)
{
    json_t root;
    root["presets"] = json_t::array();
    for (const auto& preset : presets)
        root["presets"].push_back(serializePresetJson(preset));
    return root;
}

bool deserializePresetCollectionJson(const json_t& root, std::vector<nbl::core::CCameraPreset>& presets)
{
    if (!root.contains("presets") || !root["presets"].is_array())
        return false;

    std::vector<nbl::core::CCameraPreset> loadedPresets;
    loadedPresets.reserve(root["presets"].size());
    for (const auto& entry : root["presets"])
    {
        nbl::core::CCameraPreset preset;
        nbl::system::deserializePresetJson(entry, preset);
        loadedPresets.emplace_back(std::move(preset));
    }

    presets = std::move(loadedPresets);
    return true;
}

namespace nbl::system
{

bool writeGoal(std::ostream& out, const core::CCameraGoal& goal, const int indent)
{
    if (!out)
        return false;

    out << serializeGoalJson(goal).dump(indent);
    return static_cast<bool>(out);
}

bool readGoal(std::istream& in, core::CCameraGoal& goal)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    nbl::system::deserializeGoalJson(root, goal);
    return true;
}

bool saveGoalToFile(ISystem& system, const path& filePath, const core::CCameraGoal& goal, const int indent)
{
    std::ostringstream out;
    if (!writeGoal(out, goal, indent))
        return false;
    return CCameraFileUtilities::writeTextFile(system, filePath, out.str());
}

bool loadGoalFromFile(ISystem& system, const path& filePath, core::CCameraGoal& goal)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text))
        return false;

    std::istringstream in(text);
    return readGoal(in, goal);
}

bool writePreset(std::ostream& out, const core::CCameraPreset& preset, const int indent)
{
    if (!out)
        return false;

    out << serializePresetJson(preset).dump(indent);
    return static_cast<bool>(out);
}

bool readPreset(std::istream& in, core::CCameraPreset& preset)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    nbl::system::deserializePresetJson(root, preset);
    return true;
}

bool savePresetToFile(ISystem& system, const path& filePath, const core::CCameraPreset& preset, const int indent)
{
    std::ostringstream out;
    if (!writePreset(out, preset, indent))
        return false;
    return CCameraFileUtilities::writeTextFile(system, filePath, out.str());
}

bool loadPresetFromFile(ISystem& system, const path& filePath, core::CCameraPreset& preset)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text))
        return false;

    std::istringstream in(text);
    return readPreset(in, preset);
}

bool writeKeyframeTrack(std::ostream& out, const core::CCameraKeyframeTrack& track, const int indent)
{
    if (!out)
        return false;

    out << serializeKeyframeTrackJson(track).dump(indent);
    return static_cast<bool>(out);
}

bool readKeyframeTrack(std::istream& in, core::CCameraKeyframeTrack& track)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    return deserializeKeyframeTrackJson(root, track);
}

bool saveKeyframeTrackToFile(ISystem& system, const path& filePath, const core::CCameraKeyframeTrack& track, const int indent)
{
    std::ostringstream out;
    if (!writeKeyframeTrack(out, track, indent))
        return false;
    return CCameraFileUtilities::writeTextFile(system, filePath, out.str());
}

bool loadKeyframeTrackFromFile(ISystem& system, const path& filePath, core::CCameraKeyframeTrack& track)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text))
        return false;

    std::istringstream in(text);
    return readKeyframeTrack(in, track);
}

bool writePresetCollection(std::ostream& out, std::span<const core::CCameraPreset> presets, const int indent)
{
    if (!out)
        return false;

    out << serializePresetCollectionJson(presets).dump(indent);
    return static_cast<bool>(out);
}

bool readPresetCollection(std::istream& in, std::vector<core::CCameraPreset>& presets)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    return deserializePresetCollectionJson(root, presets);
}

bool savePresetCollectionToFile(ISystem& system, const path& filePath, std::span<const core::CCameraPreset> presets, const int indent)
{
    std::ostringstream out;
    if (!writePresetCollection(out, presets, indent))
        return false;
    return CCameraFileUtilities::writeTextFile(system, filePath, out.str());
}

bool loadPresetCollectionFromFile(ISystem& system, const path& filePath, std::vector<core::CCameraPreset>& presets)
{
    std::string text;
    if (!CCameraFileUtilities::readTextFile(system, filePath, text))
        return false;

    std::istringstream in(text);
    return readPresetCollection(in, presets);
}

} // namespace nbl::system
