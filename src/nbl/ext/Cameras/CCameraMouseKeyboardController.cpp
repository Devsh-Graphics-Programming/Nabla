// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraMouseKeyboardController.hpp"

#include <algorithm>

namespace nbl::ext::cameras
{

SCameraControls CCameraMouseKeyboardController::collect(
    const SFrameWindow& window,
    const std::span<const ui::SKeyboardEvent> keyboardEvents,
    const std::span<const ui::SMouseEvent> mouseEvents)
{
    using seconds_t = std::chrono::duration<hlsl::float64_t>;

    /// @brief What the window has left after `instant`, which is zero once the window is spent.
    const auto remainingSeconds = [&](const std::chrono::microseconds instant)
    {
        const auto clamped = window.clamp(instant);
        return window.end > clamped ? seconds_t(window.end - clamped).count() : 0.0;
    };

    // a key already down when the frame opens is down for all of it, a press adds back what the window has left
    // after it, and a release takes that remainder away again, so a press and release inside one frame leave
    // exactly the interval between them
    std::array<hlsl::float64_t, ui::EKC_COUNT> heldSeconds = {};
    const auto windowSeconds = seconds_t(window.duration()).count();
    for (uint32_t key = 0u; key < ui::EKC_COUNT; ++key)
    {
        if (m_heldKeys[key])
            heldSeconds[key] = windowSeconds;
    }

    for (const auto& event : keyboardEvents)
    {
        const uint32_t key = event.keyCode;
        if (key == ui::EKC_NONE || key >= ui::EKC_COUNT)
            continue;

        if (event.action == ui::SKeyboardEvent::ECA_PRESSED)
        {
            // a key already down repeats while it is held, and a repeat is not a new press
            if (!m_heldKeys[key])
            {
                m_heldKeys[key] = true;
                heldSeconds[key] += remainingSeconds(event.timeStamp);
            }
        }
        else if (event.action == ui::SKeyboardEvent::ECA_RELEASED)
        {
            if (m_heldKeys[key])
            {
                m_heldKeys[key] = false;
                heldSeconds[key] -= remainingSeconds(event.timeStamp);
            }
        }
    }

    SCameraControls controls = {};

    const auto accumulateMouse = [&](const ui::SMouseEvent::E_EVENT_TYPE type, const hlsl::float64_t2& delta)
    {
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            const auto& slot = binding.axes[i];

            hlsl::float64_t2 gain = hlsl::float64_t2(0.0);
            switch (type)
            {
                case ui::SMouseEvent::EET_MOVEMENT:
                {
                    // a gated axis takes relative movement only while its button is held
                    const bool allowedToMove = !slot.mouseMovementGate.has_value() || isMouseButtonHeld(slot.mouseMovementGate.value());
                    if (!allowedToMove)
                        continue;

                    gain = slot.mouseMovementGain;
                    break;
                }
                case ui::SMouseEvent::EET_SCROLL:
                    gain = slot.mouseScrollGain;
                    break;
                default:
                    continue;
            }

            const auto contribution = gain.x * delta.x + gain.y * delta.y;
            if (contribution != 0.0)
                controls.axis(cameraControlAxisFromIndex(i)) += contribution;
        }
    };

    for (const auto& event : mouseEvents)
    {
        switch (event.type)
        {
            case ui::SMouseEvent::EET_CLICK:
            {
                const uint32_t button = event.clickEvent.mouseButton;
                if (button >= ui::EMB_COUNT)
                    break;
                if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
                    m_heldMouseButtons[button] = true;
                else if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
                    m_heldMouseButtons[button] = false;
                break;
            }
            case ui::SMouseEvent::EET_MOVEMENT:
                accumulateMouse(ui::SMouseEvent::EET_MOVEMENT, hlsl::float64_t2(
                    static_cast<hlsl::float64_t>(event.movementEvent.relativeMovementX),
                    static_cast<hlsl::float64_t>(event.movementEvent.relativeMovementY)));
                break;
            case ui::SMouseEvent::EET_SCROLL:
                accumulateMouse(ui::SMouseEvent::EET_SCROLL, hlsl::float64_t2(
                    static_cast<hlsl::float64_t>(event.scrollEvent.verticalScroll),
                    static_cast<hlsl::float64_t>(event.scrollEvent.horizontalScroll)));
                break;
            default:
                break;
        }
    }

    for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
    {
        const auto& slot = binding.axes[i];
        if (slot.keyRate == 0.0)
            continue;

        hlsl::float64_t seconds = 0.0;
        if (slot.positiveKey != ui::EKC_NONE && slot.positiveKey < ui::EKC_COUNT)
            seconds += heldSeconds[slot.positiveKey];
        if (slot.negativeKey != ui::EKC_NONE && slot.negativeKey < ui::EKC_COUNT)
            seconds -= heldSeconds[slot.negativeKey];
        if (seconds != 0.0)
            controls.axis(cameraControlAxisFromIndex(i)) += seconds * slot.keyRate;
    }

    return controls;
}

SCameraControls CCameraMouseKeyboardController::collect(
    const std::chrono::microseconds nextPresentationTimestamp,
    const std::span<const ui::SKeyboardEvent> keyboardEvents,
    const std::span<const ui::SMouseEvent> mouseEvents)
{
    using seconds_t = std::chrono::duration<hlsl::float64_t>;

    // the first frame has no earlier timestamp to measure against, so no key counts as held for any length
    if (!m_hasPreviousFrame)
    {
        m_hasPreviousFrame = true;
        m_lastPresentationTimestamp = nextPresentationTimestamp;
    }

    const auto maxFrameDelta = std::chrono::duration_cast<std::chrono::microseconds>(seconds_t(std::max(maxFrameDeltaSeconds, 0.0)));
    const SFrameWindow window = {
        .start = std::max(m_lastPresentationTimestamp, nextPresentationTimestamp - maxFrameDelta),
        .end = nextPresentationTimestamp
    };
    m_lastPresentationTimestamp = nextPresentationTimestamp;

    return collect(window, keyboardEvents, mouseEvents);
}

void CCameraMouseKeyboardController::reset()
{
    m_heldKeys.reset();
    m_heldMouseButtons.reset();
    m_hasPreviousFrame = false;
}

} // namespace nbl::ext::cameras
