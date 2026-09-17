// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_PERSISTENCE_HPP_
#define _C_CAMERA_PRESET_PERSISTENCE_HPP_

#include <string>
#include <string_view>

#include "CCameraPreset.hpp"
#include "nbl/system/ISystem.h"
#include "nbl/system/path.h"

namespace nbl::ext::cameras
{


/// @brief JSON text and file helpers for goals and presets.
struct CCameraPresetPersistenceUtilities final
{
    /// @brief Serialize one camera goal to JSON text.
    static std::string serializeGoal(const CCameraGoal& goal, int indent = 2);
    /// @brief Deserialize one camera goal from JSON text.
    static bool deserializeGoal(std::string_view text, CCameraGoal& goal, std::string* error = nullptr);

    /// @brief Save one camera goal to a file.
    static bool saveGoalToFile(system::ISystem& system, const system::path& path, const CCameraGoal& goal, int indent = 2);
    /// @brief Load one camera goal from a file.
    static bool loadGoalFromFile(system::ISystem& system, const system::path& path, CCameraGoal& goal, std::string* error = nullptr);

    /// @brief Serialize one camera preset to JSON text.
    static std::string serializePreset(const CCameraPreset& preset, int indent = 2);
    /// @brief Deserialize one camera preset from JSON text.
    static bool deserializePreset(std::string_view text, CCameraPreset& preset, std::string* error = nullptr);

    /// @brief Save one camera preset to a file.
    static bool savePresetToFile(system::ISystem& system, const system::path& path, const CCameraPreset& preset, int indent = 2);
    /// @brief Load one camera preset from a file.
    static bool loadPresetFromFile(system::ISystem& system, const system::path& path, CCameraPreset& preset, std::string* error = nullptr);
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_PRESET_PERSISTENCE_HPP_
