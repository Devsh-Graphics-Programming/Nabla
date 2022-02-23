#ifndef __NBL_UI_I_INPUT_EVENT_H_INCLUDED__
#define __NBL_UI_I_INPUT_EVENT_H_INCLUDED__
#include <chrono>

namespace nbl::ui
{
class IWindow;

struct SEventBase
{
    std::chrono::microseconds timeStamp;
    SEventBase(std::chrono::microseconds ts) : timeStamp(ts) {}
};

struct SMouseEvent : SEventBase
{
    SMouseEvent(std::chrono::microseconds ts) : SEventBase(ts) {}
    enum E_EVENT_TYPE : uint8_t
    {
        EET_UNITIALIZED = 0,
        EET_CLICK = 1,
        EET_SCROLL = 2,
        EET_MOVEMENT = 4
    } type = EET_UNITIALIZED;
    struct SClickEvent
    {
        int16_t clickPosX, clickPosY;
        ui::E_MOUSE_BUTTON mouseButton;
        enum E_ACTION : uint8_t
        {
            EA_UNITIALIZED = 0,
            EA_PRESSED = 1,
            EA_RELEASED = 2
        } action = EA_UNITIALIZED;
    };
    struct SScrollEvent
    {
        int16_t verticalScroll, horizontalScroll;
    };
    struct SRelativeMovementEvent
    {
        // UNORM value
        int16_t relativeMovementX, relativeMovementY;
    };
    union
    {
        SClickEvent clickEvent;
        SScrollEvent scrollEvent;
        SRelativeMovementEvent movementEvent;
    };
    IWindow* window;
};


struct SKeyboardEvent : SEventBase
{
    SKeyboardEvent(std::chrono::microseconds ts) : SEventBase(ts) { }
    enum E_KEY_ACTION : uint8_t
    {
        ECA_UNITIALIZED = 0,
        ECA_PRESSED = 1,
        ECA_RELEASED = 2
    } action = ECA_UNITIALIZED;
    ui::E_KEY_CODE keyCode = ui::EKC_NONE;
    IWindow* window = nullptr;
};

}
#endif