// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_
#define _C_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_

#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "CCameraScriptedRuntime.hpp"
#include "CCameraSequenceScriptPersistence.hpp"

namespace nbl::system
{

class ISystem;

/// @brief Optional scripted control overrides parsed alongside one runtime payload.
struct CCameraScriptedControlOverrides
{
    bool hasKeyboardScale = false;
    float keyboardScale = 1.f;
    bool hasMouseMoveScale = false;
    float mouseMoveScale = 1.f;
    bool hasMouseScrollScale = false;
    float mouseScrollScale = 1.f;
    bool hasTranslationScale = false;
    float translationScale = 1.f;
    bool hasRotationScale = false;
    float rotationScale = 1.f;
};

/// @brief Parsed low-level scripted runtime payload plus optional compact authored sequence.
struct CCameraScriptedInputParseResult
{
    bool enabled = true;
    bool hasLog = false;
    bool log = false;
    bool hardFail = false;
    bool visualDebug = false;
    float visualTargetFps = 0.f;
    float visualCameraHoldSeconds = 0.f;
    bool hasEnableActiveCameraMovement = false;
    bool enableActiveCameraMovement = true;
    bool exclusive = false;
    std::string capturePrefix = "script";
    CCameraScriptedControlOverrides cameraControls = {};
    CCameraScriptedTimeline timeline = {};
    std::optional<core::CCameraSequenceScript> sequence;
    std::vector<std::string> warnings;
};

struct CCameraScriptedRuntimePersistenceUtilities final
{
    static inline void appendScriptedInputParseWarning(CCameraScriptedInputParseResult& out, std::string warning)
    {
        out.warnings.emplace_back(std::move(warning));
    }
};

/// @brief Parse one low-level scripted runtime payload from an existing stream.
bool readCameraScriptedInput(std::istream& in, CCameraScriptedInputParseResult& out, std::string* error = nullptr);
/// @brief Parse one low-level scripted runtime payload directly from text.
bool readCameraScriptedInput(std::string_view text, CCameraScriptedInputParseResult& out, std::string* error = nullptr);
/// @brief Load one low-level scripted runtime payload from a file.
bool loadCameraScriptedInputFromFile(ISystem& system, const path& path, CCameraScriptedInputParseResult& out, std::string* error = nullptr);

} // namespace nbl::system

#endif // _C_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_
