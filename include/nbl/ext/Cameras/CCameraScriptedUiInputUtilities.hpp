#ifndef _C_CAMERA_SCRIPTED_UI_INPUT_UTILITIES_HPP_
#define _C_CAMERA_SCRIPTED_UI_INPUT_UTILITIES_HPP_

#include <chrono>
#include <vector>

#include "CCameraScriptedRuntime.hpp"
#include "nbl/ui/SInputEvent.h"

namespace nbl::ui
{

/// @brief Convert authored scripted keyboard and mouse payloads into runtime UI input events.
///
/// The scripted runtime stores compact authoring-friendly payloads. This helper
/// expands them into the concrete `SKeyboardEvent` and `SMouseEvent` objects
/// consumed by the same input path as live window events.
struct CCameraScriptedUiInputUtilities final
{
    /// @brief Build one runtime keyboard event from authored scripted keyboard data.
    static inline SKeyboardEvent makeScriptedKeyboardEvent(
        const std::chrono::microseconds timestamp,
        IWindow* const window,
        const system::CCameraScriptedInputEvent::KeyboardData& authoredKeyboard)
    {
        SKeyboardEvent event(timestamp);
        event.keyCode = authoredKeyboard.key;
        event.action =
            authoredKeyboard.action == system::CCameraScriptedInputEvent::KeyboardData::Action::Pressed ?
            SKeyboardEvent::ECA_PRESSED :
            SKeyboardEvent::ECA_RELEASED;
        event.window = window;
        return event;
    }

    /// @brief Build one runtime mouse event from authored scripted mouse data.
    static inline bool tryBuildScriptedMouseEvent(
        const std::chrono::microseconds timestamp,
        IWindow* const window,
        const system::CCameraScriptedInputEvent::MouseData& authoredMouse,
        SMouseEvent& outEvent)
    {
        outEvent = SMouseEvent(timestamp);
        outEvent.window = window;

        switch (authoredMouse.type)
        {
            case system::CCameraScriptedInputEvent::MouseData::Type::Click:
                outEvent.type = SMouseEvent::EET_CLICK;
                outEvent.clickEvent.mouseButton = authoredMouse.button;
                outEvent.clickEvent.action =
                    authoredMouse.action == system::CCameraScriptedInputEvent::MouseData::ClickAction::Pressed ?
                    SMouseEvent::SClickEvent::EA_PRESSED :
                    SMouseEvent::SClickEvent::EA_RELEASED;
                outEvent.clickEvent.clickPosX = authoredMouse.x;
                outEvent.clickEvent.clickPosY = authoredMouse.y;
                return true;
            case system::CCameraScriptedInputEvent::MouseData::Type::Scroll:
                outEvent.type = SMouseEvent::EET_SCROLL;
                outEvent.scrollEvent.verticalScroll = authoredMouse.v;
                outEvent.scrollEvent.horizontalScroll = authoredMouse.h;
                return true;
            case system::CCameraScriptedInputEvent::MouseData::Type::Movement:
                outEvent.type = SMouseEvent::EET_MOVEMENT;
                outEvent.movementEvent.relativeMovementX = authoredMouse.dx;
                outEvent.movementEvent.relativeMovementY = authoredMouse.dy;
                return true;
            default:
                return false;
        }
    }

    /// @brief Append one authored scripted input batch to existing runtime event buffers.
    static inline void appendScriptedUiInputEvents(
        const std::chrono::microseconds timestamp,
        IWindow* const window,
        const std::vector<system::CCameraScriptedInputEvent::KeyboardData>& authoredKeyboard,
        const std::vector<system::CCameraScriptedInputEvent::MouseData>& authoredMouse,
        std::vector<SKeyboardEvent>& outKeyboard,
        std::vector<SMouseEvent>& outMouse)
    {
        outKeyboard.reserve(outKeyboard.size() + authoredKeyboard.size());
        for (const auto& keyboardEvent : authoredKeyboard)
            outKeyboard.emplace_back(makeScriptedKeyboardEvent(timestamp, window, keyboardEvent));

        outMouse.reserve(outMouse.size() + authoredMouse.size());
        for (const auto& mouseEvent : authoredMouse)
        {
            SMouseEvent builtEvent(timestamp);
            if (tryBuildScriptedMouseEvent(timestamp, window, mouseEvent, builtEvent))
                outMouse.emplace_back(builtEvent);
        }
    }
};

} // namespace nbl::ui

#endif // _C_CAMERA_SCRIPTED_UI_INPUT_UTILITIES_HPP_
