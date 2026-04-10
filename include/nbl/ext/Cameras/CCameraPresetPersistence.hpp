// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_PERSISTENCE_HPP_
#define _C_CAMERA_PRESET_PERSISTENCE_HPP_

#include <string>
#include <string_view>

#include "CCameraPreset.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

/// @brief JSON text and file helpers for goals and presets.
struct CCameraPresetPersistenceUtilities final
{
    /// @brief Serialize one camera goal to JSON text.
    static std::string serializeGoal(const core::CCameraGoal& goal, int indent = 2);
    /// @brief Deserialize one camera goal from JSON text.
    static bool deserializeGoal(std::string_view text, core::CCameraGoal& goal, std::string* error = nullptr);

    /// @brief Save one camera goal to a file.
    static bool saveGoalToFile(ISystem& system, const path& path, const core::CCameraGoal& goal, int indent = 2);
    /// @brief Load one camera goal from a file.
    static bool loadGoalFromFile(ISystem& system, const path& path, core::CCameraGoal& goal, std::string* error = nullptr);

    /// @brief Serialize one camera preset to JSON text.
    static std::string serializePreset(const core::CCameraPreset& preset, int indent = 2);
    /// @brief Deserialize one camera preset from JSON text.
    static bool deserializePreset(std::string_view text, core::CCameraPreset& preset, std::string* error = nullptr);

    /// @brief Save one camera preset to a file.
    static bool savePresetToFile(ISystem& system, const path& path, const core::CCameraPreset& preset, int indent = 2);
    /// @brief Load one camera preset from a file.
    static bool loadPresetFromFile(ISystem& system, const path& path, core::CCameraPreset& preset, std::string* error = nullptr);
};

} // namespace nbl::system

#endif // _C_CAMERA_PRESET_PERSISTENCE_HPP_
