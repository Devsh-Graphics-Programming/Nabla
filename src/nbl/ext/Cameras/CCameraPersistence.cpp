// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraPersistence.hpp"

#include <array>

#include "CCameraJsonPersistenceUtilities.hpp"
#include "nbl/ext/Cameras/CCameraSequenceScriptPersistence.hpp"
#include "nlohmann/json.hpp"

using json_t = nlohmann::json;

namespace nbl::ext::cameras
{

namespace impl
{

struct CCameraPersistenceJsonUtilities final
{
    static json_t serializeGoalJson(const CCameraGoal& goal)
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
        json["camera_capabilities"] = static_cast<uint32_t>(goal.sourceCapabilities.value);
        json["camera_goal_state_mask"] = static_cast<uint32_t>(goal.sourceGoalStateMask.value);

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

    static json_t serializePresetJson(const CCameraPreset& preset)
    {
        auto json = serializeGoalJson(CCameraPresetUtilities::makeGoalFromPreset(preset));
        json["name"] = preset.name;
        json["identifier"] = preset.identifier;
        return json;
    }

    static json_t serializeKeyframeTrackJson(const CCameraKeyframeTrack& track)
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

    static bool deserializeKeyframeTrackJson(const json_t& root, CCameraKeyframeTrack& track)
    {
        if (!root.contains("keyframes") || !root["keyframes"].is_array())
            return false;

        track = {};
        for (const auto& entry : root["keyframes"])
        {
            CCameraKeyframe keyframe;
            if (entry.contains("time"))
                keyframe.time = std::max(0.f, entry["time"].get<float>());
            CCameraJsonPersistenceUtilities::deserializePresetJson(entry, keyframe.preset);
            track.keyframes.emplace_back(std::move(keyframe));
        }

        CCameraKeyframeTrackUtilities::sortKeyframeTrackByTime(track);
        CCameraKeyframeTrackUtilities::normalizeSelectedKeyframeTrack(track);
        return true;
    }

    static json_t serializePresetCollectionJson(std::span<const CCameraPreset> presets)
    {
        json_t root;
        root["presets"] = json_t::array();
        for (const auto& preset : presets)
            root["presets"].push_back(serializePresetJson(preset));
        return root;
    }

    static bool deserializePresetCollectionJson(const json_t& root, std::vector<CCameraPreset>& presets)
    {
        if (!root.contains("presets") || !root["presets"].is_array())
            return false;

        std::vector<CCameraPreset> loadedPresets;
        loadedPresets.reserve(root["presets"].size());
        for (const auto& entry : root["presets"])
        {
            CCameraPreset preset;
            CCameraJsonPersistenceUtilities::deserializePresetJson(entry, preset);
            loadedPresets.emplace_back(std::move(preset));
        }

        presets = std::move(loadedPresets);
        return true;
    }

    template<typename Value, typename DeserializeFn>
    static bool deserializeJsonText(
        std::string_view text,
        Value& out,
        const char* invalidPayloadMessage,
        DeserializeFn&& deserializeFn,
        std::string* error)
    {
        try
        {
            const auto root = json_t::parse(text);
            if (!deserializeFn(root, out))
            {
                if (error)
                    *error = invalidPayloadMessage;
                return false;
            }
            return true;
        }
        catch (const json_t::exception& e)
        {
            if (error)
                *error = e.what();
            return false;
        }
    }

    static bool readTextFileOrSetError(nbl::system::ISystem& system, const nbl::system::path& filePath, std::string& text, std::string* error, const char* openMessage)
    {
        return CFileUtilities::readTextFile(system, filePath, text, error, openMessage);
    }
};

} // namespace impl

std::string CCameraPresetPersistenceUtilities::serializeGoal(const CCameraGoal& goal, const int indent)
{
    return impl::CCameraPersistenceJsonUtilities::serializeGoalJson(goal).dump(indent);
}

bool CCameraPresetPersistenceUtilities::deserializeGoal(std::string_view text, CCameraGoal& goal, std::string* error)
{
    return impl::CCameraPersistenceJsonUtilities::deserializeJsonText(
        text,
        goal,
        "Camera goal JSON payload is invalid.",
        [](const json_t& root, CCameraGoal& outGoal)
        {
            impl::CCameraJsonPersistenceUtilities::deserializeGoalJson(root, outGoal);
            return true;
        },
        error);
}

bool CCameraPresetPersistenceUtilities::saveGoalToFile(system::ISystem& system, const system::path& filePath, const CCameraGoal& goal, const int indent)
{
    return CFileUtilities::writeTextFile(system, filePath, serializeGoal(goal, indent));
}

bool CCameraPresetPersistenceUtilities::loadGoalFromFile(system::ISystem& system, const system::path& filePath, CCameraGoal& goal, std::string* error)
{
    std::string text;
    if (!impl::CCameraPersistenceJsonUtilities::readTextFileOrSetError(system, filePath, text, error, "Cannot open camera goal file."))
        return false;

    return deserializeGoal(text, goal, error);
}

std::string CCameraPresetPersistenceUtilities::serializePreset(const CCameraPreset& preset, const int indent)
{
    return impl::CCameraPersistenceJsonUtilities::serializePresetJson(preset).dump(indent);
}

bool CCameraPresetPersistenceUtilities::deserializePreset(std::string_view text, CCameraPreset& preset, std::string* error)
{
    return impl::CCameraPersistenceJsonUtilities::deserializeJsonText(
        text,
        preset,
        "Camera preset JSON payload is invalid.",
        [](const json_t& root, CCameraPreset& outPreset)
        {
            impl::CCameraJsonPersistenceUtilities::deserializePresetJson(root, outPreset);
            return true;
        },
        error);
}

bool CCameraPresetPersistenceUtilities::savePresetToFile(system::ISystem& system, const system::path& filePath, const CCameraPreset& preset, const int indent)
{
    return CFileUtilities::writeTextFile(system, filePath, serializePreset(preset, indent));
}

bool CCameraPresetPersistenceUtilities::loadPresetFromFile(system::ISystem& system, const system::path& filePath, CCameraPreset& preset, std::string* error)
{
    std::string text;
    if (!impl::CCameraPersistenceJsonUtilities::readTextFileOrSetError(system, filePath, text, error, "Cannot open camera preset file."))
        return false;

    return deserializePreset(text, preset, error);
}

std::string CCameraKeyframeTrackPersistenceUtilities::serializeKeyframeTrack(const CCameraKeyframeTrack& track, const int indent)
{
    return impl::CCameraPersistenceJsonUtilities::serializeKeyframeTrackJson(track).dump(indent);
}

bool CCameraKeyframeTrackPersistenceUtilities::deserializeKeyframeTrack(std::string_view text, CCameraKeyframeTrack& track, std::string* error)
{
    return impl::CCameraPersistenceJsonUtilities::deserializeJsonText(
        text,
        track,
        "Camera keyframe track JSON payload is invalid.",
        [](const json_t& root, CCameraKeyframeTrack& outTrack)
        {
            return impl::CCameraPersistenceJsonUtilities::deserializeKeyframeTrackJson(root, outTrack);
        },
        error);
}

bool CCameraKeyframeTrackPersistenceUtilities::saveKeyframeTrackToFile(system::ISystem& system, const system::path& filePath, const CCameraKeyframeTrack& track, const int indent)
{
    return CFileUtilities::writeTextFile(system, filePath, serializeKeyframeTrack(track, indent));
}

bool CCameraKeyframeTrackPersistenceUtilities::loadKeyframeTrackFromFile(system::ISystem& system, const system::path& filePath, CCameraKeyframeTrack& track, std::string* error)
{
    std::string text;
    if (!impl::CCameraPersistenceJsonUtilities::readTextFileOrSetError(system, filePath, text, error, "Cannot open camera keyframe track file."))
        return false;

    return deserializeKeyframeTrack(text, track, error);
}

std::string CCameraPersistenceUtilities::serializePresetCollection(std::span<const CCameraPreset> presets, const int indent)
{
    return impl::CCameraPersistenceJsonUtilities::serializePresetCollectionJson(presets).dump(indent);
}

bool CCameraPersistenceUtilities::deserializePresetCollection(std::string_view text, std::vector<CCameraPreset>& presets, std::string* error)
{
    return impl::CCameraPersistenceJsonUtilities::deserializeJsonText(
        text,
        presets,
        "Camera preset collection JSON payload is invalid.",
        [](const json_t& root, std::vector<CCameraPreset>& outPresets)
        {
            return impl::CCameraPersistenceJsonUtilities::deserializePresetCollectionJson(root, outPresets);
        },
        error);
}

bool CCameraPersistenceUtilities::savePresetCollectionToFile(system::ISystem& system, const system::path& filePath, std::span<const CCameraPreset> presets, const int indent)
{
    return CFileUtilities::writeTextFile(system, filePath, serializePresetCollection(presets, indent));
}

bool CCameraPersistenceUtilities::loadPresetCollectionFromFile(system::ISystem& system, const system::path& filePath, std::vector<CCameraPreset>& presets, std::string* error)
{
    std::string text;
    if (!impl::CCameraPersistenceJsonUtilities::readTextFileOrSetError(system, filePath, text, error, "Cannot open camera preset collection file."))
        return false;

    return deserializePresetCollection(text, presets, error);
}

} // namespace nbl::ext::cameras
