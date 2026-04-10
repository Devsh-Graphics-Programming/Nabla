#ifndef _NBL_EXAMPLES_CAMERA_JSON_PERSISTENCE_UTILITIES_HPP_INCLUDED_
#define _NBL_EXAMPLES_CAMERA_JSON_PERSISTENCE_UTILITIES_HPP_INCLUDED_

#include <array>

#include "nbl/ext/Cameras/CCameraFileUtilities.hpp"
#include "nbl/ext/Cameras/CCameraGoal.hpp"
#include "nbl/ext/Cameras/CCameraPresetFlow.hpp"
#include "nlohmann/json.hpp"

namespace nbl::system
{

template<typename Json>
inline void deserializeGoalJson(const Json& entry, core::CCameraGoal& goal)
{
    goal = {};

    if (entry.contains("camera_kind"))
        goal.sourceKind = static_cast<core::ICamera::CameraKind>(entry["camera_kind"].get<uint32_t>());
    if (entry.contains("camera_capabilities"))
        goal.sourceCapabilities = entry["camera_capabilities"].get<uint32_t>();
    if (entry.contains("camera_goal_state_mask"))
        goal.sourceGoalStateMask = entry["camera_goal_state_mask"].get<uint32_t>();

    if (entry.contains("position") && entry["position"].is_array())
    {
        const auto values = entry["position"].get<std::array<double, 3>>();
        goal.position = hlsl::float64_t3(values[0], values[1], values[2]);
    }
    if (entry.contains("orientation") && entry["orientation"].is_array())
    {
        const auto values = entry["orientation"].get<std::array<hlsl::float64_t, 4>>();
        goal.orientation = hlsl::CCameraMathUtilities::makeQuaternionFromComponents<hlsl::float64_t>(values[0], values[1], values[2], values[3]);
    }
    if (entry.contains("target_position") && entry["target_position"].is_array())
    {
        const auto values = entry["target_position"].get<std::array<double, 3>>();
        goal.targetPosition = hlsl::float64_t3(values[0], values[1], values[2]);
        goal.hasTargetPosition = true;
    }
    if (entry.contains("distance"))
    {
        goal.distance = entry["distance"].get<float>();
        goal.hasDistance = true;
    }
    if (entry.contains("orbit_u"))
    {
        goal.orbitUv.x = entry["orbit_u"].get<double>();
        goal.hasOrbitState = true;
    }
    if (entry.contains("orbit_v"))
    {
        goal.orbitUv.y = entry["orbit_v"].get<double>();
        goal.hasOrbitState = true;
    }
    if (entry.contains("orbit_distance"))
    {
        goal.orbitDistance = entry["orbit_distance"].get<float>();
        goal.hasOrbitState = true;
    }
    if (entry.contains("path_s") && entry.contains("path_u") && entry.contains("path_v"))
    {
        goal.pathState.s = entry["path_s"].get<double>();
        goal.pathState.u = entry["path_u"].get<double>();
        goal.pathState.v = entry["path_v"].get<double>();
        goal.pathState.roll = entry.contains("path_roll") ? entry["path_roll"].get<double>() : 0.0;
        goal.hasPathState = true;
    }
    if (entry.contains("dynamic_base_fov"))
    {
        goal.dynamicPerspectiveState.baseFov = entry["dynamic_base_fov"].get<float>();
        goal.hasDynamicPerspectiveState = true;
    }
    if (entry.contains("dynamic_reference_distance"))
    {
        goal.dynamicPerspectiveState.referenceDistance = entry["dynamic_reference_distance"].get<float>();
        goal.hasDynamicPerspectiveState = true;
    }
}

template<typename Json>
inline void deserializePresetJson(const Json& entry, core::CCameraPreset& preset)
{
    preset = {};
    if (entry.contains("name"))
        preset.name = entry["name"].get<std::string>();
    if (entry.contains("identifier"))
        preset.identifier = entry["identifier"].get<std::string>();

    core::CCameraGoal goal = {};
    deserializeGoalJson(entry, goal);
    core::CCameraPresetUtilities::assignGoalToPreset(preset, goal);
}

} // namespace nbl::system

#endif // _NBL_EXAMPLES_CAMERA_JSON_PERSISTENCE_UTILITIES_HPP_INCLUDED_
