// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_PERSISTENCE_HPP_
#define _C_CAMERA_PRESET_PERSISTENCE_HPP_

#include <iosfwd>

#include "CCameraPreset.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

/// @brief Serialize one camera goal into an existing stream.
bool writeGoal(std::ostream& out, const core::CCameraGoal& goal, int indent = 2);
/// @brief Deserialize one camera goal from an existing stream.
bool readGoal(std::istream& in, core::CCameraGoal& goal);

/// @brief Save one camera goal to a file.
bool saveGoalToFile(ISystem& system, const path& path, const core::CCameraGoal& goal, int indent = 2);
/// @brief Load one camera goal from a file.
bool loadGoalFromFile(ISystem& system, const path& path, core::CCameraGoal& goal);

/// @brief Serialize one camera preset into an existing stream.
bool writePreset(std::ostream& out, const core::CCameraPreset& preset, int indent = 2);
/// @brief Deserialize one camera preset from an existing stream.
bool readPreset(std::istream& in, core::CCameraPreset& preset);

/// @brief Save one camera preset to a file.
bool savePresetToFile(ISystem& system, const path& path, const core::CCameraPreset& preset, int indent = 2);
/// @brief Load one camera preset from a file.
bool loadPresetFromFile(ISystem& system, const path& path, core::CCameraPreset& preset);

} // namespace nbl::system

#endif // _C_CAMERA_PRESET_PERSISTENCE_HPP_
