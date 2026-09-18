// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_CAMERAS_C_CAMERA_MOUSE_KEYBOARD_PRESETS_HPP_
#define _NBL_EXT_CAMERAS_C_CAMERA_MOUSE_KEYBOARD_PRESETS_HPP_

#include "CCameraKindUtilities.hpp"
#include "CCameraMouseKeyboardController.hpp"

namespace nbl::ext::cameras
{

/// @brief Default mouse/keyboard bindings per camera kind, and edits over a binding.
struct CCameraMouseKeyboardPresets final
{
    /// @brief Rates and gains the default bindings use.
    ///
    /// An axis carries either a length, in world units, or an angle, in radians, so there are two of each.
    struct SDefaults final
    {
        /// @brief World units per second a key is held.
        static inline constexpr hlsl::float64_t KeyboardLengthRate = 20.0;
        /// @brief Radians per second a key is held.
        static inline constexpr hlsl::float64_t KeyboardAngleRate = 3.0;
        /// @brief World units per count of relative mouse movement.
        static inline constexpr hlsl::float64_t MouseLengthGain = 0.01;
        /// @brief Radians per count of relative mouse movement.
        static inline constexpr hlsl::float64_t MouseAngleGain = 0.003;
        /// @brief World units per scroll step.
        static inline constexpr hlsl::float64_t ScrollLengthGain = 1.0;
        /// @brief Radians per scroll step.
        static inline constexpr hlsl::float64_t ScrollAngleGain = 0.1;
    };

    /// @brief Default binding for one camera kind; no gate on any slot. `Unknown` gives an empty binding.
    static SCameraMouseKeyboardBinding makeDefaultBinding(ICamera::CameraKind kind);
    static SCameraMouseKeyboardBinding makeDefaultBinding(const ICamera& camera);
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_C_CAMERA_MOUSE_KEYBOARD_PRESETS_HPP_
