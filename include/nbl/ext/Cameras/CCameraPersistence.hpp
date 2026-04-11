// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PERSISTENCE_HPP_
#define _C_CAMERA_PERSISTENCE_HPP_

#include <string>
#include <string_view>
#include <span>
#include <vector>

#include "CCameraKeyframeTrackPersistence.hpp"
#include "CCameraPresetPersistence.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

struct CCameraPersistenceUtilities final
{
    /// @brief Serialize a preset collection to JSON text.
    static std::string serializePresetCollection(std::span<const core::CCameraPreset> presets, int indent = 2);
    /// @brief Parse a preset collection from JSON text.
    static bool deserializePresetCollection(std::string_view text, std::vector<core::CCameraPreset>& presets, std::string* error = nullptr);

    /// @brief Save a preset collection to disk as JSON.
    static bool savePresetCollectionToFile(ISystem& system, const path& path, std::span<const core::CCameraPreset> presets, int indent = 2);
    /// @brief Load a preset collection from disk.
    static bool loadPresetCollectionFromFile(ISystem& system, const path& path, std::vector<core::CCameraPreset>& presets, std::string* error = nullptr);
};

} // namespace nbl::system

#endif // _C_CAMERA_PERSISTENCE_HPP_
