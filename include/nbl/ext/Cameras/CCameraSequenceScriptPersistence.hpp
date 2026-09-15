// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SEQUENCE_SCRIPT_PERSISTENCE_HPP_
#define _C_CAMERA_SEQUENCE_SCRIPT_PERSISTENCE_HPP_

#include <string>
#include <string_view>

#include "CCameraSequenceScript.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

struct CCameraSequenceScriptPersistenceUtilities final
{
    /// @brief Parse one compact camera-sequence script directly from JSON text.
    static bool deserializeCameraSequenceScript(std::string_view text, core::CCameraSequenceScript& out, std::string* error = nullptr);
    /// @brief Load one compact camera-sequence script from a file.
    static bool loadCameraSequenceScriptFromFile(ISystem& system, const path& path, core::CCameraSequenceScript& out, std::string* error = nullptr);
};

} // namespace nbl::system

#endif // _C_CAMERA_SEQUENCE_SCRIPT_PERSISTENCE_HPP_
