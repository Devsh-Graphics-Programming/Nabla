// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_HPP_
#define _C_CAMERA_PRESET_HPP_

#include <span>
#include <string>

#include "CCameraGoal.hpp"

namespace nbl::core
{

/// @brief Named persisted camera state built on top of `CCameraGoal`.
struct CCameraPreset
{
    std::string name;
    std::string identifier;
    CCameraGoal goal = {};
};

/// @brief Time-stamped preset entry used by playback and authoring tools.
struct CCameraKeyframe
{
    CCameraPreset preset;
    float time = 0.f;
};

struct CCameraPresetUtilities final
{
    static inline void assignGoalToPreset(CCameraPreset& preset, const CCameraGoal& goal)
    {
        preset.goal = CCameraGoalUtilities::canonicalizeGoal(goal);
    }

    static inline CCameraGoal makeGoalFromPreset(const CCameraPreset& preset)
    {
        return CCameraGoalUtilities::canonicalizeGoal(preset.goal);
    }

    /// @brief Compare two named presets through their shared canonical goal state.
    static inline bool comparePresets(const CCameraPreset& lhs, const CCameraPreset& rhs,
        const double posEps, const double rotEpsDeg, const double scalarEps)
    {
        return lhs.name == rhs.name &&
            lhs.identifier == rhs.identifier &&
            CCameraGoalUtilities::compareGoals(makeGoalFromPreset(lhs), makeGoalFromPreset(rhs), posEps, rotEpsDeg, scalarEps);
    }

    /// @brief Compare two preset collections element-by-element through the shared canonical goal state.
    static inline bool comparePresetCollections(std::span<const CCameraPreset> lhs, std::span<const CCameraPreset> rhs,
        const double posEps, const double rotEpsDeg, const double scalarEps)
    {
        if (lhs.size() != rhs.size())
            return false;

        for (size_t i = 0u; i < lhs.size(); ++i)
        {
            if (!comparePresets(lhs[i], rhs[i], posEps, rotEpsDeg, scalarEps))
                return false;
        }

        return true;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_PRESET_HPP_
