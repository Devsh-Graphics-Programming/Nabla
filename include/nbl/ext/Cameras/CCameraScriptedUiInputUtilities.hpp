#ifndef _C_CAMERA_SCRIPTED_UI_INPUT_UTILITIES_HPP_
#define _C_CAMERA_SCRIPTED_UI_INPUT_UTILITIES_HPP_

#include <chrono>
#include <vector>

#include "CCameraScriptedRuntime.hpp"
#include "nbl/ui/SInputEvent.h"

namespace nbl::ext::cameras
{

/// @brief Convert authored scripted keyboard and mouse payloads into runtime UI input events.
///
/// The scripted runtime stores compact authoring-friendly payloads. This helper
/// expands them into the concrete `ui::SKeyboardEvent` and `ui::SMouseEvent` objects
/// consumed by the same input path as live window events.
struct CCameraScriptedUiInputUtilities final
{
    /// @brief Build one runtime keyboard event from authored scripted keyboard data.
    static inline ui::SKeyboardEvent makeScriptedKeyboardEvent(
        const std::chrono::microseconds timestamp,
        ui::IWindow* const window,
        const CCameraScriptedInputEvent::KeyboardData& authoredKeyboard)
    {
        ui::SKeyboardEvent event(timestamp);
        event.keyCode = authoredKeyboard.key;
        event.action =
            authoredKeyboard.action == CCameraScriptedInputEvent::KeyboardData::Action::Pressed ?
            ui::SKeyboardEvent::ECA_PRESSED :
            ui::SKeyboardEvent::ECA_RELEASED;
        event.window = window;
        return event;
    }

    /// @brief Build one runtime mouse event from authored scripted mouse data.
    static inline bool tryBuildScriptedMouseEvent(
        const std::chrono::microseconds timestamp,
        ui::IWindow* const window,
        const CCameraScriptedInputEvent::MouseData& authoredMouse,
        ui::SMouseEvent& outEvent)
    {
        outEvent = ui::SMouseEvent(timestamp);
        outEvent.window = window;

        switch (authoredMouse.type)
        {
            case CCameraScriptedInputEvent::MouseData::Type::Click:
                outEvent.type = ui::SMouseEvent::EET_CLICK;
                outEvent.clickEvent.mouseButton = authoredMouse.button;
                outEvent.clickEvent.action =
                    authoredMouse.action == CCameraScriptedInputEvent::MouseData::ClickAction::Pressed ?
                    ui::SMouseEvent::SClickEvent::EA_PRESSED :
                    ui::SMouseEvent::SClickEvent::EA_RELEASED;
                outEvent.clickEvent.clickPosX = authoredMouse.position.x;
                outEvent.clickEvent.clickPosY = authoredMouse.position.y;
                return true;
            case CCameraScriptedInputEvent::MouseData::Type::Scroll:
                outEvent.type = ui::SMouseEvent::EET_SCROLL;
                outEvent.scrollEvent.verticalScroll = authoredMouse.scroll.x;
                outEvent.scrollEvent.horizontalScroll = authoredMouse.scroll.y;
                return true;
            case CCameraScriptedInputEvent::MouseData::Type::Movement:
                outEvent.type = ui::SMouseEvent::EET_MOVEMENT;
                outEvent.movementEvent.relativeMovementX = authoredMouse.delta.x;
                outEvent.movementEvent.relativeMovementY = authoredMouse.delta.y;
                return true;
            default:
                return false;
        }
    }

    /// @brief Append one authored scripted input batch to existing runtime event buffers.
    static inline void appendScriptedUiInputEvents(
        const std::chrono::microseconds timestamp,
        ui::IWindow* const window,
        const std::vector<CCameraScriptedInputEvent::KeyboardData>& authoredKeyboard,
        const std::vector<CCameraScriptedInputEvent::MouseData>& authoredMouse,
        std::vector<ui::SKeyboardEvent>& outKeyboard,
        std::vector<ui::SMouseEvent>& outMouse)
    {
        outKeyboard.reserve(outKeyboard.size() + authoredKeyboard.size());
        for (const auto& keyboardEvent : authoredKeyboard)
            outKeyboard.emplace_back(makeScriptedKeyboardEvent(timestamp, window, keyboardEvent));

        outMouse.reserve(outMouse.size() + authoredMouse.size());
        for (const auto& mouseEvent : authoredMouse)
        {
            ui::SMouseEvent builtEvent(timestamp);
            if (tryBuildScriptedMouseEvent(timestamp, window, mouseEvent, builtEvent))
                outMouse.emplace_back(builtEvent);
        }
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_SCRIPTED_UI_INPUT_UTILITIES_HPP_
