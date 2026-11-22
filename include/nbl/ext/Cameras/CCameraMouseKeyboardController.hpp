// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_CAMERAS_C_CAMERA_MOUSE_KEYBOARD_CONTROLLER_HPP_
#define _NBL_EXT_CAMERAS_C_CAMERA_MOUSE_KEYBOARD_CONTROLLER_HPP_

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <optional>
#include <span>

#include "nbl/core/util/bitflag.h"
#include "nbl/ui/KeyCodes.h"
#include "nbl/ui/SInputEvent.h"

#include "SCameraControls.hpp"

namespace nbl::ext::cameras
{

/// @brief How one control axis is driven by the keyboard and the mouse.
struct SMouseKeyboardAxisBinding
{
    /// @brief Key that drives the axis in the positive direction while held.
    ui::E_KEY_CODE positiveKey = ui::EKC_NONE;
    /// @brief Key that drives the axis in the negative direction while held.
    ui::E_KEY_CODE negativeKey = ui::EKC_NONE;
    /// @brief Axis units per second while one of the keys is held.
    hlsl::float64_t keyRate = 0.0;
    /// @brief Axis units per count of `relativeMovementX` and of `relativeMovementY`.
    hlsl::float64_t2 mouseMovementGain = hlsl::float64_t2(0.0);
    /// @brief Axis units per step of `verticalScroll` and of `horizontalScroll`.
    hlsl::float64_t2 mouseScrollGain = hlsl::float64_t2(0.0);
    /// @brief When set, relative movement drives the axis only while this button is held.
    std::optional<ui::E_MOUSE_BUTTON> mouseMovementGate = std::nullopt;
};

/// @brief One slot per single-bit `ECameraControlAxis`, stored by axis index.
struct SCameraMouseKeyboardBinding
{
    std::array<SMouseKeyboardAxisBinding, CameraControlAxisCount> axes = {};

    inline SMouseKeyboardAxisBinding& operator[](const ECameraControlAxis axis) { return axes[cameraControlAxisIndex(axis)]; }
    inline const SMouseKeyboardAxisBinding& operator[](const ECameraControlAxis axis) const { return axes[cameraControlAxisIndex(axis)]; }

    /// @brief Multiply the key rate and both mouse gains of every axis in `axisMask` by `scale`.
    ///
    /// Applied to a freshly made default binding this turns one speed value into absolute per-axis rates, so
    /// repeating it from a slider does not compound.
    inline void scaleSensitivity(const core::bitflag<ECameraControlAxis> axisMask, const hlsl::float64_t scale)
    {
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            if (!axisMask.hasFlags(cameraControlAxisFromIndex(i)))
                continue;

            axes[i].keyRate *= scale;
            axes[i].mouseMovementGain *= scale;
            axes[i].mouseScrollGain *= scale;
        }
    }

    /// @brief Gate relative mouse movement on `button` for every axis in `axisMask`.
    inline void setMouseMovementGate(const core::bitflag<ECameraControlAxis> axisMask, const ui::E_MOUSE_BUTTON button)
    {
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            if (axisMask.hasFlags(cameraControlAxisFromIndex(i)))
                axes[i].mouseMovementGate = button;
        }
    }

    /// @brief Drop the relative-movement gate from every axis in `axisMask`.
    inline void clearMouseMovementGate(const core::bitflag<ECameraControlAxis> axisMask)
    {
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            if (axisMask.hasFlags(cameraControlAxisFromIndex(i)))
                axes[i].mouseMovementGate = std::nullopt;
        }
    }
};

/// @brief The slice of time one frame of input covers.
///
/// The window and the timestamps of the events measured against it have to come from one clock. `nbl::ui`
/// stamps input events with `std::chrono::steady_clock` as it dispatches them, so a window built from that
/// clock is directly comparable to them.
struct SFrameWindow
{
    std::chrono::microseconds start = {};
    std::chrono::microseconds end = {};

    /// @brief How long the window is; zero when `end` precedes `start`.
    inline std::chrono::microseconds duration() const
    {
        return end > start ? end - start : std::chrono::microseconds::zero();
    }

    /// @brief `instant` brought inside the window. A window whose `end` precedes its `start` collapses onto `start`.
    inline std::chrono::microseconds clamp(const std::chrono::microseconds instant) const
    {
        return std::max(start, std::min(instant, end));
    }
};

/// @brief Turns one frame of `ui::SKeyboardEvent` and `ui::SMouseEvent` into one `SCameraControls`.
///
/// Held keys are integrated over the frame window. A key already down when the frame opens counts for all of
/// it, a press at `t` adds what the window has left after `t`, and a release at `t` takes that remainder back.
/// A key pressed and released inside one frame therefore counts the interval between the two, and any number
/// of taps in a frame sum. Event timestamps outside the window are clamped to it.
///
/// Relative mouse movement and scroll contribute their deltas times the slot gains, movement only while the
/// slot's gate button is held when one is set. Neither is measured against the clock. Events are consumed in
/// the order given.
class CCameraMouseKeyboardController
{
public:
    static inline constexpr hlsl::float64_t DefaultMaxFrameDeltaSeconds = 0.2;

    /// @brief Which keys, mouse quantities and gates drive each control axis, and at what rate. Edit between frames.
    SCameraMouseKeyboardBinding binding = {};
    /// @brief Longest window, in seconds, the presentation-timestamp overload builds.
    hlsl::float64_t maxFrameDeltaSeconds = DefaultMaxFrameDeltaSeconds;

    /// @brief Consume one frame of events over the window the caller chose.
    ///
    /// The caller owns the window and is responsible for it covering the interval its events were stamped in.
    // TODO: Needs to be revise later.
    SCameraControls collect(
        const SFrameWindow& window,
        std::span<const ui::SKeyboardEvent> keyboardEvents,
        std::span<const ui::SMouseEvent> mouseEvents);

    /// @brief Consume one frame of events over a window built from consecutive presentation timestamps.
    ///
    /// The window runs from the timestamp given to the previous call up to this one, capped at
    /// `maxFrameDeltaSeconds`. The first call, and the first after `reset()`, has no earlier timestamp and so
    /// contributes no held-key time; mouse movement and scroll still apply.
    ///
    /// TODO: a presentation timestamp is a prediction of when the frame will be shown. The oracle builds it by
    /// adding an average frame time to an instant sampled at the top of the frame, so the window this derives
    /// sits roughly one frame past the interval the events were stamped in, and events before its start collapse
    /// onto it. Both sides do come from `std::chrono::steady_clock` and so are directly comparable, which is what
    /// makes a better window possible: a caller that knows when it last drained input should build the window
    /// from that and use the overload above.
    SCameraControls collect(
        std::chrono::microseconds nextPresentationTimestamp,
        std::span<const ui::SKeyboardEvent> keyboardEvents,
        std::span<const ui::SMouseEvent> mouseEvents);

    /// @brief Forget every held key and mouse button, and start a new frame window.
    void reset();

    inline bool isKeyHeld(const ui::E_KEY_CODE key) const { return key < ui::EKC_COUNT && m_heldKeys[key]; }
    inline bool isMouseButtonHeld(const ui::E_MOUSE_BUTTON button) const { return button < ui::EMB_COUNT && m_heldMouseButtons[button]; }

private:
    std::bitset<ui::EKC_COUNT> m_heldKeys = {};
    std::bitset<ui::EMB_COUNT> m_heldMouseButtons = {};
    std::chrono::microseconds m_lastPresentationTimestamp = {};
    bool m_hasPreviousFrame = false;
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_C_CAMERA_MOUSE_KEYBOARD_CONTROLLER_HPP_
